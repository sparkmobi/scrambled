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
from uuid import uuid4
from collections import defaultdict

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

class TranscriptionRequest(BaseModel):
    urls: List[str] = []
    youtube_url: str = None

# Add this near the top of the file, after other global variables
task_results = {}
task_queue = asyncio.Queue()

def create_temp_dir():
    return tempfile.mkdtemp(dir=SCRIPT_DIR)

def cleanup_temp_files(temp_dir):
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")

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

async def transcribe_chunk(chunk, chunk_number, audio_duration, temp_dir, max_retries=5):
    temp_file_path = None
    for attempt in range(max_retries):
        try:
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix='.mp3', delete=False) as temp_file:
                temp_file_path = temp_file.name
                chunk.export(temp_file_path, format="mp3", bitrate="32k")
            
            logger.info(f"Attempting to transcribe chunk {chunk_number} (Attempt {attempt + 1}/{max_retries})")
            
            with api_key_lock:
                api_key = get_available_key(audio_duration)
            
            if api_key == "use_assemblyai":
                logger.info(f"Switching to AssemblyAI for chunk {chunk_number}")
                aai.settings.api_key = ASSEMBLYAI_API_KEY
                transcriber = aai.Transcriber()
                
                with open(temp_file_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                
                config = aai.TranscriptionConfig(language_detection=True)
                # Use run_in_executor for synchronous operations
                transcript = await asyncio.to_thread(transcriber.transcribe, audio_bytes, config)
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
                
                transcript = transcript.text
            elif api_key is None:
                raise Exception("No available API keys")
            else:
                client = AsyncGroq(api_key=api_key)
                
                with open(temp_file_path, "rb") as audio_file:
                    response = await client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3",
                        prompt="",
                        temperature=0.0,
                        response_format="text"
                    )
                
                transcript = str(response).strip()
            
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

        transcripts = []
        tasks = []
        for i, chunk in enumerate(chunks):
            audio_duration = len(chunk) / 1000
            task = asyncio.create_task(transcribe_chunk(chunk, i+1, audio_duration, temp_dir))
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

# Add this function to handle background tasks
async def process_task(task_id, task_type, data):
    try:
        if task_type == "transcribe":
            result = await transcribe_task(data)
        elif task_type == "upload":
            result = await upload_task(data)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        task_results[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        task_results[task_id] = {"status": "failed", "error": str(e)}

# Modify the transcribe function
@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    task_results[task_id] = {"status": "processing"}
    background_tasks.add_task(process_task, task_id, "transcribe", request)
    return {"task_id": task_id}

# Modify the upload_file function
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks):
    task_id = str(uuid4())
    task_results[task_id] = {"status": "processing"}
    background_tasks.add_task(process_task, task_id, "upload", file)
    return {"task_id": task_id}

# Add a new endpoint to check task results
@app.get("/task_result/{task_id}")
async def get_task_result(task_id: str):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = task_results[task_id]
    if result["status"] == "completed":
        # Optionally, remove the result from memory after it's been retrieved
        # del task_results[task_id]
        return result
    elif result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["error"])
    else:
        return {"status": "processing"}

# Implement the transcribe_task function
async def transcribe_task(request: TranscriptionRequest):
    try:
        if request.urls:
            transcriptions = []
            all_failed = True
            for url in request.urls:
                file_path, temp_dir = await download_file(url)
                if file_path:
                    try:
                        transcription = await process_audio(file_path)
                        transcriptions.append({"url": url, "transcription": transcription})
                        all_failed = False
                    finally:
                        if temp_dir:
                            await asyncio.to_thread(cleanup_temp_files, temp_dir)
                else:
                    transcriptions.append({"url": url, "error": "Failed to download the file"})
            
            if all_failed:
                raise Exception("Failed to download all provided files")
            
            return {"transcriptions": transcriptions}
        
        elif request.youtube_url:
            audio_url = await download_youtube_audio(request.youtube_url)
            if audio_url:
                file_path, temp_dir = await download_file(audio_url)
                if file_path:
                    try:
                        transcription = await process_audio(file_path)
                        return {"youtube_url": request.youtube_url, "transcription": transcription}
                    finally:
                        if temp_dir:
                            await asyncio.to_thread(cleanup_temp_files, temp_dir)
                else:
                    raise Exception("Failed to download the audio file from YouTube")
            else:
                raise Exception("Failed to get download link for YouTube video")
        
        else:
            raise Exception("No URLs or YouTube URL provided")
    
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        raise

# Implement the upload_task function
async def upload_task(file: UploadFile):
    temp_dir = create_temp_dir()
    try:
        file_extension = os.path.splitext(file.filename)[1]
        temp_file_path = os.path.join(temp_dir, f"temp_upload_{int(time.time() * 1000)}{file_extension}")
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        transcription = await process_audio(temp_file_path)
        return {"filename": file.filename, "transcription": transcription}
    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        raise
    finally:
        await asyncio.to_thread(cleanup_temp_files, temp_dir)

def signal_handler(sig, frame):
    print("Received shutdown signal. Stopping server...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    server.run()
