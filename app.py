import os
import math
import streamlit as st
import requests
import tempfile
from pydub import AudioSegment
from dotenv import load_dotenv
import logging
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Groq API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set in the environment variables")
    st.stop()

# Initialize Groq client
st.session_state.client = Groq(api_key=GROQ_API_KEY)

# Maximum chunk size (10MB in bytes)
MAX_CHUNK_SIZE = 10 * 1024 * 1024

# Get the current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def split_audio(audio_file, file_size):
    if file_size <= MAX_CHUNK_SIZE:
        return [AudioSegment.from_file(audio_file)]
    
    num_chunks = math.ceil(file_size / MAX_CHUNK_SIZE)
    audio = AudioSegment.from_file(audio_file)
    chunk_duration = len(audio) // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        end = (i + 1) * chunk_duration if i < num_chunks - 1 else len(audio)
        chunks.append(audio[start:end])
    
    return chunks

def transcribe_chunk(chunk, chunk_number):
    try:
        temp_file_path = os.path.join(SCRIPT_DIR, f"temp_chunk_{chunk_number}.wav")
        chunk.export(temp_file_path, format="wav")
        
        with open(temp_file_path, "rb") as audio_file:
            transcription = st.session_state.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                prompt="",
				temperature=0.0,
				response_format="json"
            )
        
        os.remove(temp_file_path)
        logger.info(f"Chunk {chunk_number} processed successfully")
        return transcription.text.strip()
    except Exception as e:
        logger.error(f"Error in transcribe_chunk {chunk_number}: {str(e)}")
        raise

def process_audio(file_path):
    try:
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")

        chunks = split_audio(file_path, file_size)
        logger.info(f"Audio split into {len(chunks)} chunks")

        transcripts = []
        for i, chunk in enumerate(chunks):
            try:
                transcript = transcribe_chunk(chunk, i+1)
                transcripts.append(transcript)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                transcripts.append(str(e))

        errors = [t for t in transcripts if "Error" in t]
        if errors:
            error_msgs = "\n".join(errors)
            raise Exception(f"Errors occurred during transcription:\n{error_msgs}")

        full_transcript = " ".join(t for t in transcripts if "Error" not in t)
        logger.info("All chunks processed and combined")

        return full_transcript

    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        raise

def download_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        temp_file_path = os.path.join(SCRIPT_DIR, "temp_download.mp3")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)
        return temp_file_path
    return None

st.title("Audio/Video Transcription App")

input_method = st.radio("Choose input method:", ("URL", "File Upload"))

if input_method == "URL":
    url = st.text_input("Enter the URL of the audio or video file:")
    if url:
        if st.button("Transcribe"):
            with st.spinner("Downloading and transcribing..."):
                file_path = download_file(url)
                if file_path:
                    try:
                        transcription = process_audio(file_path)
                        st.success("Transcription complete!")
                        st.text_area("Transcription:", value=transcription, height=300)
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")
                    finally:
                        os.remove(file_path)
                else:
                    st.error("Failed to download the file. Please check the URL and try again.")
else:
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file:
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                temp_file_path = os.path.join(SCRIPT_DIR, f"temp_upload.{uploaded_file.name.split('.')[-1]}")
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                try:
                    transcription = process_audio(temp_file_path)
                    st.success("Transcription complete!")
                    st.text_area("Transcription:", value=transcription, height=300)
                except Exception as e:
                    st.error(f"An error occurred during transcription: {str(e)}")
                finally:
                    os.remove(temp_file_path)

st.markdown("Note: This app supports audio files up to 25 MB in size.")