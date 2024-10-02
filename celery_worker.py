from celery import Celery
from fastapi_app import process_audio, download_file, download_youtube_audio, cleanup_temp_files
import asyncio
import logging
import os
import shutil

celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
celery_app.conf.broker_connection_retry_on_startup = True

@celery_app.task(name='tasks.transcribe_urls')
def transcribe_urls(urls):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_transcribe_urls(urls))

@celery_app.task(name='tasks.transcribe_youtube')
def transcribe_youtube(youtube_url):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_transcribe_youtube(youtube_url))

@celery_app.task(name='tasks.transcribe_file')
def transcribe_file(file_path, filename):
    loop = asyncio.get_event_loop()
    try:
        return loop.run_until_complete(_transcribe_file(file_path, filename))
    finally:
        cleanup_temp_files(file_path)  # Clean up the file, not the directory

async def _transcribe_urls(urls):
    transcriptions = []
    for url in urls:
        file_path, temp_dir = await download_file(url)
        if file_path:
            try:
                transcription = await process_audio(file_path)
                transcriptions.append({"url": url, "transcription": transcription})
            finally:
                if temp_dir:
                    cleanup_temp_files(temp_dir)
        else:
            transcriptions.append({"url": url, "error": "Failed to download the file"})
    return transcriptions

async def _transcribe_youtube(youtube_url):
    audio_url = await download_youtube_audio(youtube_url)
    if audio_url:
        file_path, temp_dir = await download_file(audio_url)
        if file_path:
            try:
                transcription = await process_audio(file_path)
                return {"youtube_url": youtube_url, "transcription": transcription}
            finally:
                if temp_dir:
                    cleanup_temp_files(temp_dir)
        else:
            return {"youtube_url": youtube_url, "error": "Failed to download the audio file from YouTube"}
    else:
        return {"youtube_url": youtube_url, "error": "Failed to get download link for YouTube video"}

async def _transcribe_file(file_path, filename):
    try:
        transcription = await process_audio(file_path)
        return {"filename": filename, "transcription": transcription}
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return {"filename": filename, "error": "File not found"}
    except Exception as e:
        logging.error(f"Error processing file {filename}: {str(e)}")
        return {"filename": filename, "error": str(e)}

def cleanup_temp_files(path):
    try:
        if os.path.isfile(path):
            os.remove(path)
            logging.info(f"Cleaned up temporary file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            logging.info(f"Cleaned up temporary directory: {path}")
        else:
            logging.warning(f"Path does not exist or is neither file nor directory: {path}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary path {path}: {str(e)}")

if __name__ == '__main__':
    celery_app.start()
