import os
import sys
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from groq import Groq, AsyncGroq
import ffmpeg
from api_key_manager import get_available_key, reset_counters, logger as api_key_logger
import time
import tempfile
import shutil
import random
import threading
import assemblyai as aai
from helper import download_youtube_audio
import asyncio
import uvicorn
import signal
import sys
from celery import Celery
from celery.result import AsyncResult

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Maximum chunk size (10MB in bytes)
MAX_CHUNK_SIZE = 10 * 1024 * 1024

# Get the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add this at the beginning of the file
file_lock = threading.Lock()
api_key_lock = threading.Lock()

# Add this to your environment variables or directly in the code (be cautious with API keys)
ASSEMBLYAI_API_KEY = "7c5d242d606542268916c235daa26031"

app = FastAPI()

# Create Celery app
celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
celery_app.conf.broker_connection_retry_on_startup = True

class TranscriptionRequest(BaseModel):
    urls: List[str] = []
    youtube_url: str = None

def create_temp_dir():
    return tempfile.mkdtemp(dir=SCRIPT_DIR)

def cleanup_temp_files(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"Cleaned up temporary file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"Cleaned up temporary directory: {path}")
        else:
            logger.warning(f"Path does not exist or is neither file nor directory: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary path {path}: {str(e)}")

def convert_opus_to_mp3(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='libmp3lame', b='128k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        logger.info(f"Converted OPUS to MP3: {output_file}")
        return output_file
    except ffmpeg.Error as e:
        logger.error(f"Error converting OPUS to MP3: {e.stderr.decode()}")
        raise

def preprocess_audio(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='libmp3lame', ac=1, ar='16k', b='32k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        preprocessed_size = os.path.getsize(output_file)
        logger.info(f"Preprocessed audio file: {output_file}")
        logger.info(f"Preprocessed file size: {preprocessed_size / (1024 * 1024):.2f} MB")
        
        return output_file
    except ffmpeg.Error as e:
        logger.error(f"Error preprocessing audio: {e.stderr.decode()}")
        raise

def split_audio(audio_file, max_duration=1750):
    audio = AudioSegment.from_mp3(audio_file)
    chunks = []
    for i in range(0, len(audio), max_duration * 1000):
        chunk = audio[i:i + max_duration * 1000]
        chunks.append(chunk)
    return chunks

def exponential_backoff(attempt, max_delay=60):
    return min(2 ** attempt + random.uniform(0, 1), max_delay)

# Initialize the AssemblyAI transcriber
aai.settings.api_key = ASSEMBLYAI_API_KEY
transcriber = aai.Transcriber()

async def detect_audio_language(file_path):
    try:
        config = aai.TranscriptionConfig(
            audio_end_at=60000,  # first 60 seconds (in milliseconds)
            language_detection=True,
            speech_model=aai.SpeechModel.nano,
        )
        transcript = await asyncio.to_thread(transcriber.transcribe, file_path, config=config)
        detected_language = transcript.detected_language_code
        logger.info(f"Detected language code: {detected_language}")
        
        # Map the detected language to 'en' or 'ar'
        if detected_language == 'en':
            return 'en'
        elif detected_language in ['ar', 'arb']:  # 'arb' is Modern Standard Arabic
            return 'ar'
        else:
            logger.warning(f"Unsupported language detected: {detected_language}. Defaulting to English.")
            return 'en'
    except Exception as e:
        logger.error(f"Error detecting audio language: {str(e)}")
        # Default to English if detection fails
        return 'en'

# Update the process_audio function to use the new detect_audio_language
async def process_audio(file_path):
    temp_dir = create_temp_dir()
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.opus':
            logger.info("Detected OPUS file. Converting to MP3...")
            mp3_file = os.path.join(temp_dir, f"converted_audio_{int(time.time() * 1000)}.mp3")
            file_path = convert_opus_to_mp3(file_path, mp3_file)

        input_size = os.path.getsize(file_path)
        logger.info(f"Input file size: {input_size / (1024 * 1024):.2f} MB")

        preprocessed_file = os.path.join(temp_dir, f"preprocessed_audio_{int(time.time() * 1000)}.mp3")
        preprocess_audio(file_path, preprocessed_file)
        
        chunks = split_audio(preprocessed_file)
        logger.info(f"Audio split into {len(chunks)} chunks")

        # Detect the language before preprocessing
        language = await detect_audio_language(file_path)
        logger.info(f"Detected language: {language}")

        transcripts = []
        tasks = []
        for i, chunk in enumerate(chunks):
            audio_duration = len(chunk) / 1000
            task = asyncio.create_task(transcribe_chunk(chunk, i+1, audio_duration, temp_dir, language))
            tasks.append(task)

        transcripts = await asyncio.gather(*tasks)

        errors = [t for t in transcripts if isinstance(t, str) and t.startswith("Error in chunk")]
        if errors:
            error_msgs = "\n".join(errors)
            raise Exception(f"Errors occurred during transcription:\n{error_msgs}")

        full_transcript = " ".join(t for t in transcripts if not (isinstance(t, str) and t.startswith("Error in chunk")))
        logger.info("All chunks processed and combined")
        logger.info(f"Full transcript (first 100 characters): {full_transcript[:100]}...")

        return full_transcript

    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        raise
    finally:
        cleanup_temp_files(temp_dir)

async def download_file(url):
    temp_dir = create_temp_dir()
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_extension = os.path.splitext(url)[1] or '.mp3'  # Default to .mp3 if no extension
            temp_file_path = os.path.join(temp_dir, f"temp_download_{int(time.time() * 1000)}{file_extension}")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(response.content)
            return temp_file_path, temp_dir
        return None, temp_dir
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        cleanup_temp_files(temp_dir)
        return None, None

async def download_youtube_audio(youtube_url):
    url = "https://youtube-mp3-downloader2.p.rapidapi.com/ytmp3/ytmp3/long_video.php"
    
    querystring = {"url": youtube_url}
    
    headers = {
        "x-rapidapi-key": "08624f09acmsh93345251cb25c60p12d48djsn5ceebe4ca205",
        "x-rapidapi-host": "youtube-mp3-downloader2.p.rapidapi.com"
    }
    
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(requests.get, url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'finished':
                return data['dlink']
            elif data['status'] == 'processing':
                await asyncio.sleep(retry_delay)
            else:
                raise HTTPException(status_code=500, detail=f"YouTube download failed: {data['status']}")
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=f"Failed to download YouTube audio: {str(e)}")
            await asyncio.sleep(retry_delay)
    
    raise HTTPException(status_code=500, detail="Max retries reached for YouTube download")

@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    try:
        if request.urls:
            task = celery_app.send_task('tasks.transcribe_urls', args=[request.urls])
            return JSONResponse(content={"task_id": task.id})
        elif request.youtube_url:
            task = celery_app.send_task('tasks.transcribe_youtube', args=[request.youtube_url])
            return JSONResponse(content={"task_id": task.id})
        else:
            raise HTTPException(status_code=400, detail="No URLs or YouTube URL provided")
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_dir = create_temp_dir()
    try:
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(temp_dir, f"temp_upload_{int(time.time() * 1000)}{file_extension}")
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        task = celery_app.send_task('tasks.transcribe_file', args=[temp_file_path, file.filename])
        return JSONResponse(content={"task_id": task.id})
    except Exception as e:
        logger.error(f"An error occurred during file upload: {str(e)}")
        cleanup_temp_files(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.ready():
        if task_result.successful():
            return JSONResponse(content={"status": "completed", "result": task_result.result})
        else:
            return JSONResponse(content={"status": "failed", "error": str(task_result.result)})
    else:
        return JSONResponse(content={"status": "processing"})

def signal_handler(sig, frame):
    print("Received shutdown signal. Stopping server...")
    sys.exit(0)

# Modify the transcribe_chunk function to use the detected language
async def transcribe_chunk(chunk, chunk_number, audio_duration, temp_dir, language):
    temp_file_path = None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.mp3', delete=False) as temp_file:
                temp_file_path = temp_file.name
                chunk.export(temp_file_path, format="mp3", bitrate="32k")
            
            logger.info(f"Attempting to transcribe chunk {chunk_number} (Attempt {attempt + 1}/{max_retries})")
            
            with api_key_lock:
                api_key = get_available_key(audio_duration)
            
            if api_key == "use_assemblyai" or attempt == max_retries - 1:
                logger.info(f"Using AssemblyAI for chunk {chunk_number}")
                aai.settings.api_key = ASSEMBLYAI_API_KEY
                
                config = aai.TranscriptionConfig(language_code=language)
                transcriber = aai.Transcriber()
                transcript = await asyncio.to_thread(transcriber.transcribe, temp_file_path, config=config)
                
                if transcript.status == "error":
                    raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
                
                transcript_text = transcript.text
            elif api_key is None:
                raise Exception("No available API keys")
            else:
                try:
                    client = AsyncGroq(api_key=api_key)
                    
                    with open(temp_file_path, "rb") as audio_file:
                        response = await client.audio.transcriptions.create(
                            file=audio_file,
                            model="whisper-large-v3",
                            prompt="",
                            temperature=0.0,
                            response_format="text",
                            language=language  # Add the detected language here
                        )
                    
                    logger.info(f"Groq API response type: {type(response)}")
                    logger.info(f"Groq API response: {response}")
                    
                    if isinstance(response, str):
                        transcript_text = response.strip()
                    elif isinstance(response, dict) and 'text' in response:
                        transcript_text = response['text'].strip()
                    elif hasattr(response, 'text'):
                        transcript_text = response.text.strip()
                    else:
                        transcript_text = str(response).strip()
                    
                except Exception as e:
                    logger.error(f"Error with Groq API: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.error(f"Error args: {e.args}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise
            
            logger.info(f"Transcribed text: {transcript_text[:100]}...")  # Log first 100 characters
            logger.info(f"Chunk {chunk_number} processed successfully")
            return transcript_text
        except Exception as e:
            logger.error(f"Error in transcribe_chunk {chunk_number} (Attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(exponential_backoff(attempt))
            else:
                raise
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file {temp_file_path} deleted")
                except Exception as e:
                    logger.error(f"Error deleting temporary file {temp_file_path}: {str(e)}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import multiprocessing

    # Determine the number of worker processes
    workers = multiprocessing.cpu_count()
    
    # Configure Uvicorn to use multiple workers
    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=8000,
        workers=workers,
        loop="asyncio",
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
