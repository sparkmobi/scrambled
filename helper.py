import os
import re
from datetime import datetime
from pytube import YouTube


def save_as_md(file_path: str, content: str) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)


def sanitize_filename(filename):
    # Remove file extension
    filename = os.path.splitext(filename)[0]

    # Replace any character that's not lowercase alphanumeric or dash with a dash
    sanitized = re.sub(r"[^a-z0-9-]", "-", filename.lower())

    # Remove leading and trailing dashes
    sanitized = sanitized.strip("-")

    # Replace multiple consecutive dashes with a single dash
    sanitized = re.sub(r"-+", "-", sanitized)

    return sanitized


def extract_filename(filepath):
    # Get the base name (file name with extension)
    base_name = os.path.basename(filepath)

    # Split the base name and extension
    file_name = os.path.splitext(base_name)[0]

    return base_name, file_name


def return_youtube_id(url: str):
    """
    Returns YouTube ID of the video.

    Args:
        url: youtube video link
    Returns:
        str: youtube id
    """
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None


def get_youtube_title(url: str):
    """
    Get the title of a YouTube video.

    Args:
        url: youtube video link
    Returns:
        str: video title
    """
    yt = YouTube(url)
    return yt.title


def download_youtube_audio(dir_path: str, url: str):
    try:
        youtube_id = return_youtube_id(url)
        if not youtube_id:
            raise ValueError("Invalid YouTube URL")

        video_title = get_youtube_title(url)
        if not video_title:
            raise ValueError("Could not retrieve video title")

        sanitized_title = sanitize_filename(video_title)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{sanitized_title}_{timestamp}.mp3"
        output_file_path = os.path.join(dir_path, output_filename)

        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_stream.download(output_path=dir_path, filename=output_filename)

        return output_file_path

    except Exception as ex:
        return None


if __name__ == "__main__":
    # Example usage
    print(download_youtube_audio("data/audio", "https://youtu.be/VCwk0Xk1oR0"))
