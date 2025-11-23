# backend/tasks.py
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os
import logging

sched = BackgroundScheduler()
sched.start()

logger = logging.getLogger(__name__)


def _get_twilio_client():
    """Create Twilio client lazily to avoid import / init troubles in module import time."""
    try:
        from twilio.rest import Client as TwilioClient
    except Exception:
        return None
    SID = os.environ.get("TWILIO_SID")
    TOKEN = os.environ.get("TWILIO_TOKEN")
    if not SID or not TOKEN:
        return None
    return TwilioClient(SID, TOKEN)


def schedule_reminder(reminder_id: str, run_at: datetime, payload: dict):
    """
    Schedule a one-off reminder job at run_at (datetime).
    Job will attempt to call Twilio to play payload['voiceUrl'] if Twilio credentials present.
    """

    def job():
        try:
            twilio_client = _get_twilio_client()
            twilio_from = os.environ.get("TWILIO_FROM")
            if not twilio_client or not twilio_from:
                logger.warning("Twilio not configured: skipping phone call for reminder %s", reminder_id)
                return

            target = payload.get("primary") or payload.get("to")
            if not target:
                logger.warning("No phone number for reminder %s", reminder_id)
                return

            voice_url = payload.get("voiceUrl")
            if voice_url:
                # Use Twilio outgoing call that plays remote audio using <Play>
                twiml = f'<Response><Play>{voice_url}</Play></Response>'
                twilio_client.calls.create(
                    to=target,
                    from_=twilio_from,
                    twiml=twiml
                )
            else:
                # fallback: use Twilio TTS via <Say>
                body = payload.get("message") or "This is your scheduled reminder."
                twiml = f'<Response><Say>{body}</Say></Response>'
                twilio_client.calls.create(
                    to=target,
                    from_=twilio_from,
                    twiml=twiml
                )
            logger.info("Reminder call attempted for %s", reminder_id)
        except Exception as e:
            logger.exception("Error in scheduled reminder job for %s: %s", reminder_id, e)

    # ensure unique job id: if job exists remove first
    try:
        existing = sched.get_job(reminder_id)
        if existing:
            existing.remove()
    except Exception:
        pass

    sched.add_job(job, "date", run_date=run_at, id=reminder_id)
    logger.info("Scheduled reminder job %s at %s", reminder_id, run_at.isoformat())
