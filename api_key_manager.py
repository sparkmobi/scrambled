import os
from supabase import create_client, Client
from datetime import datetime, timedelta
import time
import logging

# Initialize Supabase client
url: str = "https://deqoekrxwvziwmclcthu.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRlcW9la3J4d3Z6aXdtY2xjdGh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2Nzg2NjQ5OTksImV4cCI6MTk5NDI0MDk5OX0.9UvqJRTRSiY99vyEfGrJ3wSLtI3ZbwRj07BaGbE9HM4"
supabase: Client = create_client(url, key)

# Define model-specific constants
MODELS = {
    "whisper-large-v3": {
        "table_name": "api_keys",
        "minute_limit": 20,
        "day_limit": 2000,
        "hour_audio_limit": 7200,
        "day_audio_limit": 28800
    },
    "distil-whisper-large-v3-en": {
        "table_name": "api_keys_distil",
        "minute_limit": 30,  # Example: adjust these limits as needed
        "day_limit": 3000,
        "hour_audio_limit": 10800,
        "day_audio_limit": 43200
    }
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_key(audio_duration, model="whisper-large-v3", max_retries=5, retry_delay=60):
    model_config = MODELS[model]
    table_name = model_config["table_name"]
    
    for attempt in range(max_retries):
        now = datetime.now()
        
        # Reset counters if necessary
        reset_counters(model)
        
        # Log all API keys before filtering
        all_keys = supabase.table(table_name).select("*").execute()
        logger.info(f"Total API keys for {model}: {len(all_keys.data)}")
        for key in all_keys.data:
            logger.info(f"Key ID: {key['id']}, hour_audio: {key['hour_audio']}, minute_count: {key['minute_count']}, day_count: {key['day_count']}, day_audio: {key['day_audio']}")
        
        # Query for available keys
        query = supabase.table(table_name).select("*")\
            .lt("minute_count", model_config["minute_limit"])\
            .lt("day_count", model_config["day_limit"])\
            .lt("hour_audio", model_config["hour_audio_limit"] - audio_duration)\
            .lt("day_audio", model_config["day_audio_limit"] - audio_duration)\
            .order("hour_audio")
        
        logger.info(f"Query parameters: minute_count < {model_config['minute_limit']}, day_count < {model_config['day_limit']}, "
                    f"hour_audio < {model_config['hour_audio_limit'] - audio_duration}, "
                    f"day_audio < {model_config['day_audio_limit'] - audio_duration}")
        
        response = query.execute()
        
        logger.info(f"Available keys: {len(response.data)}")
        for key in response.data:
            logger.info(f"Key ID: {key['id']}, hour_audio: {key['hour_audio']}, minute_count: {key['minute_count']}, day_count: {key['day_count']}, day_audio: {key['day_audio']}")
        
        if len(response.data) > 0:
            # Get the first available key (with lowest hour_audio usage)
            key = response.data[0]
            logger.info(f"Selected key ID: {key['id']}, hour_audio: {key['hour_audio']}")
            
            # Update the usage counts
            supabase.table(table_name).update({
                "minute_count": key["minute_count"] + 1,
                "day_count": key["day_count"] + 1,
                "hour_audio": key["hour_audio"] + audio_duration,
                "day_audio": key["day_audio"] + audio_duration,
                "last_used": now.isoformat()
            }).eq("id", key["id"]).execute()
            
            return key["api_key"]
        
        if attempt < max_retries - 1:
            logger.warning(f"No available API keys. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}")
            time.sleep(retry_delay)
        else:
            logger.warning("Max retries reached. Switching to AssemblyAI.")
            return "use_assemblyai"
    
    return None

def reset_counters(model):
    model_config = MODELS[model]
    table_name = model_config["table_name"]
    now = datetime.now()
    
    # Reset minute counts
    minute_reset = supabase.table(table_name).update({"minute_count": 0})\
        .lt("last_used", (now - timedelta(seconds=30)).isoformat())\
        .execute()
    logger.info(f"Minute count reset for {len(minute_reset.data)} keys in {table_name}")
    
    # Reset hour audio
    hour_reset = supabase.table(table_name).update({"hour_audio": 0})\
        .lt("last_used", (now - timedelta(minutes=30)).isoformat())\
        .execute()
    logger.info(f"Hour audio reset for {len(hour_reset.data)} keys in {table_name}")
    
    # Reset day counts and audio
    today = now.date()
    day_reset = supabase.table(table_name).update({
        "day_count": 0, 
        "day_audio": 0,
        "last_used": now.isoformat()  # Update last_used to current time
    }).lt("last_used", today.isoformat())\
        .execute()
    logger.info(f"Day count and audio reset for {len(day_reset.data)} keys in {table_name}")

    # Log all API keys after reset
    all_keys = supabase.table(table_name).select("*").execute()
    logger.info("API keys after reset:")
    for key in all_keys.data:
        logger.info(f"Key ID: {key['id']}, hour_audio: {key['hour_audio']}, minute_count: {key['minute_count']}, day_count: {key['day_count']}, day_audio: {key['day_audio']}, last_used: {key['last_used']}")
