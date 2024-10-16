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
    "whisper-large-v3-turbo": {
        "table_name": "api_keys_distil",
        "minute_limit": 20,
        "day_limit": 2000,
        "hour_audio_limit": 7200,
        "day_audio_limit": 28800
    },
    "whisper-large-v3": {
        "table_name": "api_keys",
        "minute_limit": 20,
        "day_limit": 2000,
        "hour_audio_limit": 7200,
        "day_audio_limit": 28800
    }
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_supabase_available():
    try:
        # Try to make a simple query to Supabase
        supabase.table("api_keys").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase is not available: {str(e)}")
        return False

def get_available_key(audio_duration, max_retries=5, retry_delay=60):
    if not is_supabase_available():
        logger.warning("Supabase is not available. Falling back to AssemblyAI.")
        return "use_assemblyai", None

    models_to_try = ["whisper-large-v3-turbo", "whisper-large-v3"]

    for model in models_to_try:
        logger.info(f"Trying to get an available key for model: {model}")
        key, model_used = try_get_key(audio_duration, model, max_retries, retry_delay)
        if key != "use_assemblyai":
            return key, model_used

    logger.warning("No available keys for any model. Falling back to AssemblyAI.")
    return "use_assemblyai", None

def try_get_key(audio_duration, model, max_retries, retry_delay):
    model_config = MODELS[model]
    table_name = model_config["table_name"]
    
    for attempt in range(max_retries):
        try:
            now = datetime.now()
            
            # Reset counters if necessary
            reset_counters(model)
            
            # Query for available keys
            query = supabase.table(table_name).select("*")\
                .lt("minute_count", model_config["minute_limit"])\
                .lt("day_count", model_config["day_limit"])\
                .lt("hour_audio", model_config["hour_audio_limit"] - audio_duration)\
                .lt("day_audio", model_config["day_audio_limit"] - audio_duration)\
                .order("hour_audio")
            
            response = query.execute()
            
            if len(response.data) > 0:
                key = response.data[0]
                
                # Update the usage counts
                supabase.table(table_name).update({
                    "minute_count": key["minute_count"] + 1,
                    "day_count": key["day_count"] + 1,
                    "hour_audio": key["hour_audio"] + audio_duration,
                    "day_audio": key["day_audio"] + audio_duration,
                    "last_used": now.isoformat()
                }).eq("id", key["id"]).execute()
                
                return key["api_key"], model
            
            if attempt < max_retries - 1:
                logger.warning(f"No available API keys for {model}. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
            else:
                logger.warning(f"Max retries reached for {model}. Moving to next model or falling back to AssemblyAI.")
                return "use_assemblyai", None
        
        except Exception as e:
            logger.error(f"Error retrieving API key for {model}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.warning(f"Max retries reached due to errors for {model}. Moving to next model or falling back to AssemblyAI.")
                return "use_assemblyai", None
    
    return "use_assemblyai", None

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
