# notifications.py
import logging
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor
import telegram
from twilio.rest import Client

from config import (
    EMAIL_CONFIG, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    WHATSAPP_URL, WHATSAPP_ACCOUNT_ID, WHATSAPP_TO, WHATSAPP_MSG_TYPE
)

# Initialize Telegram bot
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

def send_sms(message_body):
    """Sends an SMS using the Twilio API."""
    logging.info("Sending SMS notification...")
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            from_=TWILIO_FROM_NUMBER,
            body=message_body,
            to=TWILIO_TO_NUMBER
        )
        return message.sid
    except Exception as e:
        logging.error(f"Failed to send SMS: {e}")
        return None

def send_message_http(message, msg_type):
    """Sends a message via HTTP POST (for WhatsApp notifications)."""
    payload = {
        "accountId": WHATSAPP_ACCOUNT_ID,
        "to": WHATSAPP_TO,
        "message": message,
        "type": msg_type
    }
    try:
        response = requests.post(WHATSAPP_URL, json=payload)
        if response.status_code == 200:
            logging.info("WhatsApp message sent successfully")
            return response.json()
        else:
            logging.error(
                f"Failed to send WhatsApp message. Status Code: {response.status_code}, Response: {response.text}")
            return {"error": response.text}
    except Exception as e:
        logging.error(f"Exception occurred while sending WhatsApp message: {str(e)}")
        return {"error": str(e)}

def send_custom_message(message):
    """Sends a custom message via WhatsApp."""
    return send_message_http(message, WHATSAPP_MSG_TYPE)

def send_email(subject, body):
    """Sends an email using SMTP."""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG["sender_email"]
    msg['To'] = EMAIL_CONFIG["receiver_email"]
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["password"])
        server.sendmail(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["receiver_email"], msg.as_string())
        server.quit()
        return "Email sent successfully"
    except Exception as e:
        return f"Failed to send email: {e}"

def send_notifications(alert_msg, image_path):
    """
    Sends notifications via SMS, WhatsApp, Telegram, and Email concurrently.
    """
    responses = {}

    def sms_task():
        sid = send_sms(alert_msg)
        responses["sms"] = sid
        logging.info(f"SMS sent with SID: {sid}")

    def whatsapp_task():
        res = send_custom_message(alert_msg)
        responses["whatsapp"] = res
        logging.info(f"WhatsApp response: {res}")

    def telegram_task():
        try:
            res = bot.send_message(chat_id=TELEGRAM_CHAT_ID,
                                   text=alert_msg + f" (Frame saved at {image_path})")
            responses["telegram"] = res
            logging.info(f"Telegram response: {res}")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
            responses["telegram"] = None

    def email_task():
        subject = "Violation Alert Notification"
        body = alert_msg + f"\nFrame saved at: {image_path}"
        res = send_email(subject, body)
        responses["email"] = res
        logging.info(f"Email response: {res}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(sms_task)
        executor.submit(whatsapp_task)
        executor.submit(telegram_task)
        executor.submit(email_task)

    return responses