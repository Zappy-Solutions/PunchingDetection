# config.py
import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.warning("dotenv not found, ensure environment variables are set externally")

logging.info("Environment variables:", dict(os.environ))

# Detection and processing settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.80))
LINE_THRESHOLD = int(os.getenv("LINE_THRESHOLD", 20))
FRAME_SKIP = int(os.getenv("FRAME_SKIP", 5))
VIOLATION_DELAY = int(os.getenv("VIOLATION_DELAY", 30))

# Video stream settings
WINDOW_WIDTH = int(os.getenv("WINDOW_WIDTH", 1280))
WINDOW_HEIGHT = int(os.getenv("WINDOW_HEIGHT", 720))
FPS = int(os.getenv("FPS", 20))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", 1000))

# RTSP and other input settings
# Hikvision
HIKVISION_IP = os.getenv("HIKVISION_IP")
HIKVISION_USERNAME = os.getenv("HIKVISION_USERNAME")
HIKVISION_PASSWORD = os.getenv("HIKVISION_PASSWORD")
HIKVISION_URL = os.getenv("HIKVISION_URL")
HIKVISION_RTSP_URL = f"rtsp://{HIKVISION_USERNAME}:{HIKVISION_PASSWORD}@{HIKVISION_IP}:554/{HIKVISION_URL}"

# Impact
IMPACT_IP = os.getenv("IMPACT_IP")
IMPACT_USERNAME = os.getenv("IMPACT_USERNAME")
IMPACT_PASSWORD = os.getenv("IMPACT_PASSWORD")
IMPACT_URL= os.getenv("IMPACT_URL")
IMPACT_RTSP_URL = f"rtsp://{IMPACT_USERNAME}:{IMPACT_PASSWORD}@{IMPACT_IP}:554/{IMPACT_URL}"

# Email configuration
EMAIL_CONFIG = {
    "sender_email": os.getenv("EMAIL_SENDER"),
    "receiver_email": os.getenv("EMAIL_RECEIVER"),
    "password": os.getenv("EMAIL_PASSWORD"),
}

# Twilio SMS configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")

# Telegram configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# WhatsApp configuration
WHATSAPP_URL = os.getenv("WHATSAPP_URL", "http://192.168.68.112:5525/api/v1/sendMessage")
WHATSAPP_ACCOUNT_ID = os.getenv("WHATSAPP_ACCOUNT_ID")
WHATSAPP_TO = os.getenv("WHATSAPP_TO")
WHATSAPP_MSG_TYPE = os.getenv("WHATSAPP_MSG_TYPE", "attendance")