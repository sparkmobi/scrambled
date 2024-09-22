import os
from supabase import create_client, Client
from datetime import datetime, timedelta
import time
import logging

# Initialize Supabase client
url: str = "https://deqoekrxwvziwmclcthu.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRlcW9la3J4d3Z6aXdtY2xjdGh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2Nzg2NjQ5OTksImV4cCI6MTk5NDI0MDk5OX0.9UvqJRTRSiY99vyEfGrJ3wSLtI3ZbwRj07BaGbE9HM4"
supabase: Client = create_client(url, key)

# API limits for whisper-large-v3
MINUTE_LIMIT = 20
DAY_LIMIT = 2000
HOUR_AUDIO_LIMIT = 7200  # seconds
DAY_AUDIO_LIMIT = 28800  # seconds

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_key(audio_duration):
    now = datetime.now()
    
    # Reset counters if necessary
    reset_counters()
    
    # Query for available keys
    query = supabase.table("api_keys").select("*")\
        .lt("minute_count", MINUTE_LIMIT)\
        .lt("day_count", DAY_LIMIT)\
        .lt("hour_audio", HOUR_AUDIO_LIMIT - audio_duration)\
        .lt("day_audio", DAY_AUDIO_LIMIT - audio_duration)\
        .order("hour_audio")
    
    logger.info(f"Query parameters: minute_count < {MINUTE_LIMIT}, day_count < {DAY_LIMIT}, "
                f"hour_audio < {HOUR_AUDIO_LIMIT - audio_duration}, "
                f"day_audio < {DAY_AUDIO_LIMIT - audio_duration}")
    
    response = query.execute()
    
    logger.info(f"Available keys: {len(response.data)}")
    for key in response.data:
        logger.info(f"Key ID: {key['id']}, hour_audio: {key['hour_audio']}, minute_count: {key['minute_count']}, day_count: {key['day_count']}, day_audio: {key['day_audio']}")
    
    if len(response.data) == 0:
        logger.warning("No available API keys")
        return None
    
    # Get the first available key (with lowest hour_audio usage)
    key = response.data[0]
    logger.info(f"Selected key ID: {key['id']}, hour_audio: {key['hour_audio']}")
    
    # Update the usage counts
    supabase.table("api_keys").update({
        "minute_count": key["minute_count"] + 1,
        "day_count": key["day_count"] + 1,
        "hour_audio": key["hour_audio"] + audio_duration,
        "day_audio": key["day_audio"] + audio_duration,
        "last_used": now.isoformat()
    }).eq("id", key["id"]).execute()
    
    return key["api_key"]

def reset_counters():
    now = datetime.now()
    
    # Reset minute counts
    minute_reset = supabase.table("api_keys").update({"minute_count": 0})\
        .lt("last_used", (now - timedelta(minutes=1)).isoformat())\
        .execute()
    logger.info(f"Minute count reset for {len(minute_reset.data)} keys")
    
    # Reset hour audio
    hour_reset = supabase.table("api_keys").update({"hour_audio": 0})\
        .lt("last_used", (now - timedelta(hours=1)).isoformat())\
        .execute()
    logger.info(f"Hour audio reset for {len(hour_reset.data)} keys")
    
    # Reset day counts and audio
    day_reset = supabase.table("api_keys").update({"day_count": 0, "day_audio": 0})\
        .lt("last_used", (now - timedelta(days=1)).isoformat())\
        .execute()
    logger.info(f"Day count and audio reset for {len(day_reset.data)} keys")
