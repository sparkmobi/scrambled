import os
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Maximum chunk size (10MB in bytes)
MAX_CHUNK_SIZE = 10 * 1024 * 1024

# Get the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

def transcribe_chunk(chunk, chunk_number, audio_duration, temp_dir):
    timestamp = int(time.time() * 1000)  # Current time in milliseconds
    temp_file_path = os.path.join(temp_dir, f"temp_{timestamp}_{chunk_number}.mp3")
    chunk.export(temp_file_path, format="mp3", bitrate="32k")
    
    try:
        # Get an available API key
        logger.info(f"Requesting API key for chunk {chunk_number} with duration {audio_duration} seconds")
        api_key = get_available_key(audio_duration)
        if api_key is None:
            raise Exception("No available API keys after maximum retries")
        
        logger.info(f"Retrieved API key for chunk {chunk_number}")
        
        # Initialize Groq client with the new API key
        client = Groq(api_key=api_key)
        
        with open(temp_file_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                prompt="",
                temperature=0.0,
                response_format="text"
            )
        
        logger.info(f"Chunk {chunk_number} processed successfully")
        
        return str(response).strip()
    except Exception as e:
        logger.error(f"Error in transcribe_chunk {chunk_number}: {str(e)}")
        raise
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

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
        temp_files = []
        for i, chunk in enumerate(chunks):
            try:
                audio_duration = len(chunk) / 1000  # Convert milliseconds to seconds
                transcript = transcribe_chunk(chunk, i+1, audio_duration, temp_dir)
                logger.info(f"Transcript for chunk {i+1}: {transcript[:50]}...")
                transcripts.append(transcript)
                progress = (i + 1) / len(chunks)
                st.progress(progress)

                # Add a delay between requests to avoid overwhelming the API
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    logger.error(f"API response: {e.response.json()}")
                raise  # Re-raise the exception to stop processing if a chunk fails

        # Calculate total size of all chunks
        total_chunk_size = sum(os.path.getsize(f) for f in temp_files if os.path.exists(f))
        logger.info(f"Total size of all chunks: {total_chunk_size / (1024 * 1024):.2f} MB")

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
    url = st.text_input("Enter the URL of the audio or video file:")
    if url:
        if st.button("Transcribe"):
            with st.spinner("Downloading and transcribing..."):
                file_path, temp_dir = download_file(url)
                if file_path:
                    try:
                        transcription = process_audio(file_path)
                        if transcription:
                            st.success("Transcription complete!")
                            st.text_area("Transcription:", value=transcription, height=300)
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
                    finally:
                        if temp_dir:
                            cleanup_temp_files(temp_dir)
                else:
                    st.error("Failed to download the file. Please check the URL and try again.")
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
