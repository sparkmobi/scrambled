import os
from supabase import create_client, Client
from datetime import datetime, timedelta
import time

# Initialize Supabase client
url: str = "https://deqoekrxwvziwmclcthu.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRlcW9la3J4d3Z6aXdtY2xjdGh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2Nzg2NjQ5OTksImV4cCI6MTk5NDI0MDk5OX0.9UvqJRTRSiY99vyEfGrJ3wSLtI3ZbwRj07BaGbE9HM4"
supabase: Client = create_client(url, key)

# API limits for whisper-large-v3
MINUTE_LIMIT = 20
DAY_LIMIT = 2000
HOUR_AUDIO_LIMIT = 7200  # seconds
DAY_AUDIO_LIMIT = 28800  # seconds

def get_available_key(audio_duration):
    now = datetime.now()
    
    # Query for an available key
    response = supabase.table("api_keys").select("*")\
        .lt("minute_count", MINUTE_LIMIT)\
        .lt("day_count", DAY_LIMIT)\
        .lt("hour_audio", HOUR_AUDIO_LIMIT - audio_duration)\
        .lt("day_audio", DAY_AUDIO_LIMIT - audio_duration)\
        .execute()
    
    if len(response.data) == 0:
        return None
    
    # Get the first available key
    key = response.data[0]
    
    # Update the usage counts
    supabase.table("api_keys").update({
        "minute_count": key["minute_count"] + 1,
        "day_count": key["day_count"] + 1,
        "hour_audio": key["hour_audio"] + audio_duration,
        "day_audio": key["day_audio"] + audio_duration,
        "last_used": now.isoformat()
    }).eq("id", key["id"]).execute()
    
    return key["api_key"]

def reset_minute_counts():
    supabase.table("api_keys").update({"minute_count": 0}).execute()

def reset_hour_audio():
    supabase.table("api_keys").update({"hour_audio": 0}).execute()

def reset_day_counts():
    supabase.table("api_keys").update({
        "day_count": 0,
        "day_audio": 0
    }).execute()
