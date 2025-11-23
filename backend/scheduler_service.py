# backend/scheduler_service.py

import os
import time
import requests
import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from supabase import create_client, Client

# -------------------------
# Supabase Init
# -------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Timezone (India)
IST = pytz.timezone("Asia/Kolkata")
scheduler = BackgroundScheduler(timezone=IST)

# Flask backend URL
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")


# ============================================================
# Helpers
# ============================================================

def _normalize_time_str(t: str):
    """
    Accepts strings like "2:24", "02:24", "2:24 PM" and normalizes to "HH:MM".
    Returns None if parsing fails.
    """
    if not t:
        return None
    try:
        raw = str(t).strip()
        parts = raw.split(":")
        if len(parts) < 2:
            return None

        h = "".join([c for c in parts[0] if c.isdigit()])
        m = "".join([c for c in parts[1] if c.isdigit()])

        if not h or not m:
            return None

        h = int(h)
        m = int(m)

        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None

        return f"{h:02d}:{m:02d}"
    except Exception:
        return None


def _normalize_day_code(day_str: str):
    """
    Converts things like "Saturday", "sat", "SAT" -> "sat"
    """
    if not day_str:
        return None
    return str(day_str).strip().lower()[:3]


# ============================================================
# Scheduler Core
# ============================================================

def check_and_trigger_reminders():
    """
    Runs every minute. Fetches reminders from Supabase table `reminders`.

    Expected table structure (recommended):
      - id         (uuid)
      - user_key   (text)
      - created_at (double precision or timestamptz)
      - payload    (jsonb)

    New data is stored inside payload, e.g.:
      payload = {
        "time": "HH:MM",
        "days": [...],
        "enabled": true,
        "user_mobile": "...",
        "member_mobile": "...",
        "voice_url": "...",
        "last_triggered": <timestamp>,
        ...
      }

    For backward compatibility we also look at top-level "time" and "enabled"
    if present.
    """

    now = datetime.datetime.now(IST)
    current_time = now.strftime("%H:%M")
    current_day = now.strftime("%a").lower()[:3]

    print(f"[Scheduler] Tick @ {now.strftime('%Y-%m-%d %H:%M:%S %Z')} "
          f"(time={current_time}, day={current_day})")

    try:
        # Fetch all reminders
        res = supabase.table("reminders").select("*").execute()
        reminders = res.data or []

        if not reminders:
            print("[Scheduler] No reminders found.")
            return

        print(f"[Scheduler] Loaded {len(reminders)} reminder rows.")

        triggered_count = 0

        for rem in reminders:
            try:
                reminder_id = rem.get("id")
                user_key = rem.get("user_key") or "anon"
                payload = rem.get("payload") or {}

                # 🔹 Prefer values inside payload (new schema), fallback to top-level
                raw_time = (
                    payload.get("time")
                    or payload.get("customTime")
                    or payload.get("reminder_time")
                    or rem.get("time")
                )

                enabled = rem.get("enabled")
                if enabled is None:
                    enabled = payload.get("enabled", True)

                print(
                    f"[Scheduler] Checking reminder {reminder_id} for {user_key} | "
                    f"raw_time={repr(raw_time)}, enabled={enabled}, days={payload.get('days')}, "
                    f"last_triggered={payload.get('last_triggered')}"
                )

                should_fire, reason = should_trigger_reminder_with_reason(
                    raw_time, enabled, payload, current_time, current_day
                )

                if should_fire:
                    print(f"[Scheduler] 🚀 Triggering reminder {reminder_id} ({reason})")
                    trigger_reminder_call(user_key, reminder_id, payload)

                    # Mark last_triggered in payload and update row
                    new_payload = dict(payload)
                    new_payload["last_triggered"] = time.time()

                    supabase.table("reminders") \
                        .update({"payload": new_payload}) \
                        .eq("id", reminder_id) \
                        .execute()

                    triggered_count += 1
                else:
                    print(f"[Scheduler] Skipped {reminder_id}: {reason}")

            except Exception as e:
                print(f"[Scheduler] Error processing reminder row {rem.get('id')}: {e}")

        print(f"[Scheduler] Done. Triggered: {triggered_count}")

    except Exception as e:
        print(f"[Scheduler] Fatal error in scheduler: {e}")


def should_trigger_reminder_with_reason(raw_time, enabled, payload, current_time, current_day):
    """
    Returns (bool, reason_string):
      - bool: whether this reminder should fire now
      - reason: logging information
    """

    # 1) Enabled?
    if not enabled:
        return False, "disabled"

    # 2) Time check
    norm_time = _normalize_time_str(raw_time)
    if not norm_time:
        return False, f"invalid reminder time={repr(raw_time)}"
    if norm_time != current_time:
        return False, f"time mismatch ({norm_time} != {current_time})"

    # 3) Day-of-week check
    days = payload.get("days")
    if isinstance(days, list) and len(days) > 0:
        norm_days = [_normalize_day_code(d) for d in days if _normalize_day_code(d)]
        if current_day not in norm_days:
            return False, f"day mismatch ({current_day} not in {norm_days})"
    # If days is missing/invalid, treat as every day

    # 4) last_triggered to prevent duplicates
    last = payload.get("last_triggered", 0)
    try:
        last = float(last or 0)
    except Exception:
        last = 0

    if last:
        diff = time.time() - last
        if diff < 120:  # 2 minutes
            return False, f"already triggered {int(diff)}s ago"

    return True, "all checks passed"


# ============================================================
# Trigger Call
# ============================================================

def trigger_reminder_call(user_key, reminder_id, payload):
    try:
        user_mobile = payload.get("user_mobile") or f"+{user_key}"

        body = {
            "reminder_id": reminder_id,
            "user_mobile": user_mobile,
        }

        url = f"{BACKEND_URL.rstrip('/')}/trigger-reminder-call"
        print(f"[Scheduler] Calling POST {url} {body}")

        r = requests.post(url, json=body, timeout=30)

        if r.status_code != 200:
            print(f"[Scheduler] ❌ Failed to trigger call: {r.status_code} {r.text}")
        else:
            print(f"[Scheduler] ✅ Triggered call for {reminder_id}")

    except Exception as e:
        print(f"[Scheduler] Error triggering call: {e}")


# ============================================================
# Start / Stop
# ============================================================

def start_scheduler():
    if scheduler.running:
        print("[Scheduler] Already running.")
        return

    scheduler.add_job(
        check_and_trigger_reminders,
        CronTrigger(minute="*", timezone=IST),
        id="check_reminders",
        replace_existing=True,
    )
    scheduler.start()
    print("[Scheduler] Started (checks every minute).")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        print("[Scheduler] Stopped.")
    else:
        print("[Scheduler] Not running.")


def test_scheduler():
    print("[Scheduler] Manual test...")
    check_and_trigger_reminders()