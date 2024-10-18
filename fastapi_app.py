import os
import sys
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
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
    youtube_url: Optional[str] = None
    timestamps: bool = Field(default=True, description="Include timestamps in the transcription")
    diarization: bool = Field(default=True, description="Enable speaker diarization")

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

async def process_audio(file_path, timestamps=True, diarization=True):
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

        # Assume English for now, you can add language detection later
        language = 'en'
        logger.info(f"Assumed language: {language}")

        transcripts = []
        tasks = []
        for i, chunk in enumerate(chunks):
            audio_duration = len(chunk) / 1000
            task = asyncio.create_task(transcribe_chunk(chunk, i+1, audio_duration, temp_dir, language, timestamps, diarization))
            tasks.append(task)

        transcripts = await asyncio.gather(*tasks)

        errors = [t for t in transcripts if isinstance(t, str) and t.startswith("Error in chunk")]
        if errors:
            error_msgs = "\n".join(errors)
            raise Exception(f"Errors occurred during transcription:\n{error_msgs}")

        full_transcript = " ".join(t["transcript"] for t in transcripts if not (isinstance(t, str) and t.startswith("Error in chunk")))
        full_transcript_json = [item for t in transcripts for item in t["transcript_json"]]

        logger.info("All chunks processed and combined")
        logger.info(f"Full transcript (first 100 characters): {full_transcript[:100]}...")

        return {
            "transcript": full_transcript,
            "transcript_json": full_transcript_json
        }

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

async def transcribe_chunk(chunk, chunk_number, audio_duration, temp_dir, language, timestamps, diarization):
    temp_file_path = None
    max_retries = 5
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.mp3', delete=False) as temp_file:
                temp_file_path = temp_file.name
                chunk.export(temp_file_path, format="mp3", bitrate="32k")
            
            logger.info(f"Attempting to transcribe chunk {chunk_number} (Attempt {attempt + 1}/{max_retries})")
            
            api_key, model = get_available_key(audio_duration)
            
            if api_key == "use_assemblyai":
                logger.info(f"Using AssemblyAI for chunk {chunk_number}")
                transcript = await use_assemblyai_transcription(temp_file_path, timestamps, diarization)
            else:
                try:
                    client = AsyncGroq(api_key=api_key)
                    logger.info(f"Using Groq model: {model} for chunk {chunk_number}")
                    transcript = await use_groq_transcription(client, temp_file_path, model, timestamps, diarization)
                    logger.info(f"Groq transcription successful for chunk {chunk_number}")
                except Exception as e:
                    logger.error(f"Error with Groq API for chunk {chunk_number}: {str(e)}")
                    logger.error(f"Error type: {type(e)}")
                    logger.error(f"Error args: {e.args}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(exponential_backoff(attempt))
                        continue
                    else:
                        logger.info(f"Falling back to AssemblyAI for chunk {chunk_number}")
                        transcript = await use_assemblyai_transcription(temp_file_path, timestamps, diarization)
            
            logger.info(f"Transcribed text for chunk {chunk_number}: {transcript['transcript'][:100]}...")  # Log first 100 characters
            logger.info(f"Chunk {chunk_number} processed successfully")
            return transcript
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

    raise Exception(f"Failed to transcribe chunk {chunk_number} after {max_retries} attempts")

async def use_assemblyai_transcription(file_path, timestamps, diarization):
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    config = aai.TranscriptionConfig(
        speaker_labels=diarization,
        auto_highlights=True,
        word_boost=[],
        boost_param="default",
        punctuate=True,
        format_text=True,
        dual_channel=False,
        webhook_url=None,
        webhook_auth_header_name=None,
        webhook_auth_header_value=None,
        audio_start_from=None,
        audio_end_at=None,
        # Remove word_timestamps parameter as it's not supported
    )
    transcriber = aai.Transcriber()
    transcript = await asyncio.to_thread(transcriber.transcribe, file_path, config=config)
    if transcript.status == "error":
        raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
    
    return format_assemblyai_response(transcript, timestamps, diarization)

def format_assemblyai_response(transcript, timestamps, diarization):
    full_transcript = transcript.text
    transcript_json = []
    for utterance in transcript.utterances:
        entry = {
            "text": utterance.text,
        }
        if timestamps:
            entry["start"] = utterance.start / 1000  # Convert to seconds
            entry["end"] = utterance.end / 1000  # Convert to seconds
            entry["duration"] = (utterance.end - utterance.start) / 1000  # Convert to seconds
        if diarization:
            entry["speaker"] = utterance.speaker
        transcript_json.append(entry)
    
    return {
        "transcript": full_transcript,
        "transcript_json": transcript_json
    }

async def use_groq_transcription(client, file_path, model, timestamps, diarization):
    with open(file_path, "rb") as audio_file:
        response = await client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            prompt="",
            temperature=0.0,
            response_format="verbose_json"
        )
    
    logger.info(f"Groq API response type: {type(response)}")
    logger.info(f"Groq API response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")
    
    return format_groq_response(response, timestamps, diarization)

def format_groq_response(response, timestamps, diarization):
    full_transcript = response.get('text', '')
    transcript_json = []
    current_time = 0
    
    segments = response.get('segments', [])
    for segment in segments:
        entry = {
            "text": segment.get('text', ''),
        }
        if timestamps:
            entry["start"] = current_time
            entry["end"] = current_time + segment.get('duration', 0)
            entry["duration"] = segment.get('duration', 0)
            current_time = entry["end"]
        if diarization and 'speaker' in segment:
            entry["speaker"] = segment.get('speaker', 'Unknown')
        transcript_json.append(entry)
    
    return {
        "transcript": full_transcript,
        "transcript_json": transcript_json
    }

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
