import os
import sys
import math
import streamlit as st
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
import logging
from groq import Groq
import ffmpeg
from api_key_manager import get_available_key, reset_counters, logger as api_key_logger
from datetime import datetime, timedelta
import time
from helper import download_youtube_audio
import tempfile
import shutil
import random
import threading
from tempfile import NamedTemporaryFile
import assemblyai as aai
from io import BytesIO

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

def create_temp_dir():
    return tempfile.mkdtemp(dir=SCRIPT_DIR)

def cleanup_temp_files(temp_dir):
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {temp_dir}: {str(e)}")

def convert_opus_to_mp3(input_file, output_file):
    """Convert OPUS file to MP3."""
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
    """Preprocess audio file to 16kHz mono MP3."""
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='libmp3lame', ac=1, ar='16k', b='32k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        
        # Get and log the size of the preprocessed file
        preprocessed_size = os.path.getsize(output_file)
        logger.info(f"Preprocessed audio file: {output_file}")
        logger.info(f"Preprocessed file size: {preprocessed_size / (1024 * 1024):.2f} MB")
        
        return output_file
    except ffmpeg.Error as e:
        logger.error(f"Error preprocessing audio: {e.stderr.decode()}")
        raise

def split_audio(audio_file, max_duration=1750):
    """Split audio into chunks of max_duration seconds."""
    audio = AudioSegment.from_mp3(audio_file)
    chunks = []
    for i in range(0, len(audio), max_duration * 1000):
        chunk = audio[i:i + max_duration * 1000]
        chunks.append(chunk)
    return chunks

def exponential_backoff(attempt, max_delay=60):
    delay = min(2 ** attempt + random.uniform(0, 1), max_delay)
    time.sleep(delay)

def transcribe_chunk(chunk, chunk_number, audio_duration, temp_dir, max_retries=5):
    temp_file_path = None
    for attempt in range(max_retries):
        try:
            # Create a new temporary file for each attempt
            with NamedTemporaryFile(dir=temp_dir, suffix='.mp3', delete=False) as temp_file:
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
                
                config = aai.TranscriptionConfig(speaker_labels=True)
                transcript = transcriber.transcribe(audio_bytes, config)
                
                if transcript.status == aai.TranscriptStatus.error:
                    raise Exception(f"AssemblyAI transcription failed: {transcript.error}")
                
                transcript = transcript.text
            elif api_key is None:
                raise Exception("No available API keys")
            else:
                client = Groq(api_key=api_key)
                
                with open(temp_file_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
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
                exponential_backoff(attempt)
            else:
                raise
        finally:
            # Delete the temporary file after each attempt
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file {temp_file_path} deleted")
                except Exception as e:
                    logger.error(f"Error deleting temporary file {temp_file_path}: {str(e)}")

def process_audio(file_path):
    temp_dir = create_temp_dir()
    try:
        # Check if the file is OPUS format
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.opus':
            logger.info("Detected OPUS file. Converting to MP3...")
            mp3_file = os.path.join(temp_dir, generate_unique_filename("converted_audio", ".mp3"))
            file_path = convert_opus_to_mp3(file_path, mp3_file)

        # Log the size of the input file
        input_size = os.path.getsize(file_path)
        logger.info(f"Input file size: {input_size / (1024 * 1024):.2f} MB")

        # Preprocess the audio file
        preprocessed_file = os.path.join(temp_dir, generate_unique_filename("preprocessed_audio", ".mp3"))
        preprocess_audio(file_path, preprocessed_file)
        
        chunks = split_audio(preprocessed_file)
        logger.info(f"Audio split into {len(chunks)} chunks")

        transcripts = []
        chunk_locks = [threading.Lock() for _ in range(len(chunks))]
        for i, chunk in enumerate(chunks):
            try:
                audio_duration = len(chunk) / 1000
                with chunk_locks[i]:
                    transcript = transcribe_chunk(chunk, i+1, audio_duration, temp_dir)
                transcripts.append(transcript)
                progress = (i + 1) / len(chunks)
                st.progress(progress)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                transcripts.append(f"Error in chunk {i+1}: {str(e)}")

        errors = [t for t in transcripts if t.startswith("Error in chunk")]
        if errors:
            error_msgs = "\n".join(errors)
            raise Exception(f"Errors occurred during transcription:\n{error_msgs}")

        full_transcript = " ".join(t for t in transcripts if not t.startswith("Error in chunk"))
        logger.info("All chunks processed and combined")
        logger.info(f"Full transcript (first 100 characters): {full_transcript[:100]}...")

        return full_transcript

    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        raise
    finally:
        cleanup_temp_files(temp_dir)

def download_file(url):
    temp_dir = create_temp_dir()
    try:
        response = requests.get(url)
        if response.status_code == 200:
            file_extension = os.path.splitext(url)[1] or '.mp3'  # Default to .mp3 if no extension
            temp_file_path = os.path.join(temp_dir, generate_unique_filename("temp_download", file_extension))
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(response.content)
            return temp_file_path, temp_dir
        return None, temp_dir
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        cleanup_temp_files(temp_dir)
        return None, None

def generate_unique_filename(prefix, extension):
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    return f"{prefix}_{timestamp}{extension}"

st.title("Audio/Video Transcription App")

input_method = st.radio("Choose input method:", ("URL", "File Upload", "YouTube URL"))

if input_method == "URL":
    urls = st.text_area("Enter the URLs of the audio or video files (one per line):")
    if urls and st.button("Transcribe"):
        urls = urls.split('\n')
        progress_bar = st.progress(0)
        for i, url in enumerate(urls):
            with st.spinner(f"Processing URL {i+1}/{len(urls)}: {url}"):
                file_path, temp_dir = download_file(url)
                if file_path:
                    try:
                        transcription = process_audio(file_path)
                        if transcription:
                            st.success(f"Transcription complete for {url}!")
                            st.text_area(f"Transcription for {url}:", value=transcription, height=300)
                    except Exception as e:
                        st.error(f"An error occurred during transcription of {url}: {str(e)}")
                    finally:
                        if temp_dir:
                            cleanup_temp_files(temp_dir)
                else:
                    st.error(f"Failed to download the file from {url}. Please check the URL and try again.")
            progress_bar.progress((i + 1) / len(urls))
elif input_method == "YouTube URL":
    youtube_url = st.text_input("Enter the YouTube URL of the video:")
    if youtube_url:
        if st.button("Transcribe"):
            with st.spinner("Downloading and transcribing..."):
                file_path = download_youtube_audio(SCRIPT_DIR, youtube_url)
                if file_path:
                    try:
                        transcription = process_audio(file_path)
                        if transcription:
                            st.success("Transcription complete!")
                            st.text_area("Transcription:", value=transcription, height=300)
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
                    finally:
                        os.remove(file_path)
                else:
                    st.error("Failed to download the YouTube video. Please check the URL and try again.")
else:
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg", "opus"])
    if uploaded_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                temp_dir = create_temp_dir()
                try:
                    file_extension = os.path.splitext(uploaded_file.name)[1]
                    temp_file_path = os.path.join(temp_dir, generate_unique_filename("temp_upload", file_extension))
                    with open(temp_file_path, "wb") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                    transcription = process_audio(temp_file_path)
                    if transcription:
                        st.success("Transcription complete!")
                        st.text_area("Transcription:", value=transcription, height=300)
                except Exception as e:
                    st.error(f"An error occurred during transcription: {str(e)}")
                finally:
                    cleanup_temp_files(temp_dir)

st.markdown("Note: This app supports audio files up to 25 MB in size. Files will be compressed to reduce size if necessary.")
