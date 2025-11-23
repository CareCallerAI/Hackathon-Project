# backend/app.py (Supabase full migration, no Firebase)

import os
import tempfile
import uuid
import time
import requests
import subprocess
from datetime import datetime

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

from google_client import (
    synthesize_text,
    analyze_sentiment,
    analyze_entities,
    translate_text,
    speech_to_text_bytes,
)
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse
from xml.sax.saxutils import escape
from urllib.parse import quote_plus

from scheduler_service import start_scheduler, stop_scheduler, test_scheduler

from supabase import create_client, Client

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==================== SUPABASE INIT ====================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "media")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ============================================================
# Helpers
# ============================================================

def normalize_time_str(t: str):
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
    except:
        return None

def normalize_user_key(mobile: str) -> str:
    return (mobile or "anon").replace("+", "").replace(" ", "")


def upload_to_storage(local_path: str, dest_path: str, content_type: str = "audio/mpeg") -> str:
    """Upload file to Supabase storage using v1-style upload (x-upsert header)."""
    with open(local_path, "rb") as f:
        supabase.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            dest_path,
            f,
            {
                "content-type": content_type,
                "x-upsert": "true",  # must be string, not bool
            },
        )
    public_url = supabase.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(dest_path)
    return public_url


# ==================== TWILIO / OPENAI CONFIG ====================

TWILIO_SID = os.environ.get("TWILIO_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")
twilio_client = TwilioClient(TWILIO_SID, TWILIO_TOKEN) if TWILIO_SID and TWILIO_TOKEN else None

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Simple in-memory OTP store for dev
otp_store = {}

# Map app language codes to Google language codes
LANG_TO_GCLOUD = {
    "en": "en-IN",
    "hi": "hi-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "mr": "mr-IN",
}


# ==================== SCHEDULER START ====================

try:
    print("⏰ Starting scheduler...")
    start_scheduler()
except Exception as e:
    print("✗ Failed to start scheduler:", e)


# ==================== SENTIMENT HELPERS ====================


def heuristic_sentiment(text: str) -> dict:
    if not text:
        return {"score": 0.0, "magnitude": 0.0}

    t = text.lower()

    positive_words = [
        "happy",
        "good",
        "great",
        "fine",
        "better",
        "okay",
        "ok",
        "relaxed",
        "peaceful",
        "comfortable",
        "well",
        "joy",
        "glad",
    ]
    negative_words = [
        "sad",
        "pain",
        "hurts",
        "hurt",
        "bad",
        "terrible",
        "worried",
        "afraid",
        "lonely",
        "tired",
        "unwell",
        "sick",
        "depressed",
        "awful",
        "horrible",
        "worst",
        "crying",
        "angry",
        "frustrated",
    ]

    score = 0.0
    magnitude = 0.0

    for w in positive_words:
        if w in t:
            score += 0.4
            magnitude += 0.4

    for w in negative_words:
        if w in t:
            score -= 0.4
            magnitude += 0.4

    score = max(-1.0, min(1.0, score))
    if magnitude == 0.0:
        return {"score": 0.0, "magnitude": 0.0}
    return {"score": score, "magnitude": magnitude}


def generate_dynamic_reply(transcript: str, mood_label: str, base_en: str) -> str:
    if not OPENAI_API_KEY:
        return base_en

    try:
        prompt = (
            "You are a very kind, empathetic health companion talking to an elderly person.\n"
            "You will be given:\n"
            f"- Their recent utterance (transcript)\n"
            f"- A mood label (very_negative, negative, neutral, positive, very_positive)\n\n"
            "Reply with a SHORT, gentle message (1–3 sentences) in simple English, "
            "without asking complex questions. Do NOT mention 'mood score' or 'sentiment'.\n\n"
            f"Transcript: {transcript or '[no text]'}\n"
            f"Mood label: {mood_label or 'unknown'}\n"
        )

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a gentle, empathetic health companion.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.5,
                "max_tokens": 120,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        text = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return text or base_en
    except Exception as e:
        print("LLM generate_dynamic_reply error:", e)
        return base_en


# ==================== AUTH / OTP ====================


@app.route("/send-otp", methods=["POST"])
def send_otp():
    data = request.json or {}
    mobile = data.get("mobile")
    if not mobile:
        return jsonify({"success": False, "message": "mobile required"}), 400

    import random

    code = str(random.randint(100000, 999999))
    otp_store[mobile] = code

    if twilio_client and TWILIO_FROM:
        try:
            twilio_client.messages.create(
                body=f"Your OTP: {code}", from_=TWILIO_FROM, to=mobile
            )
        except Exception as e:
            print("Twilio SMS error:", e)

    # Dev only: returning OTP in response
    return jsonify({"success": True, "message": "OTP sent", "otp": code})


@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json or {}
    mobile = data.get("mobile")
    code = data.get("otp")

    if otp_store.get(mobile) == code:
        key = normalize_user_key(mobile)
        try:
            supabase.table("users").upsert(
                {"user_key": key, "mobile": mobile}, on_conflict="user_key"
            ).execute()
        except Exception as e:
            print("Supabase users upsert error:", e)
        return jsonify({"success": True, "token": f"demo-token-{key}"})

    return jsonify({"success": False, "message": "invalid otp"}), 400


# ==================== SIMPLE NOTIFICATION/REMINDER (LEGACY) ====================


@app.route("/notifications", methods=["POST"])
def set_notification():
    payload = request.json or {}
    user = payload.get("user") or "u1"
    user_key = normalize_user_key(user)
    try:
        supabase.table("notifications").upsert(
            {"user_key": user_key, "payload": payload}, on_conflict="user_key"
        ).execute()
    except Exception as e:
        print("Supabase notifications upsert error:", e)
    return jsonify({"success": True})


@app.route("/reminders", methods=["POST"])
def set_reminder():
    """Legacy simple reminder endpoint (not the medication reminders table)."""

    data = request.form.to_dict() or request.json or {}
    file = request.files.get("file")

    mobile = data.get("user_mobile") or data.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    if file:
        fname = f"reminder-{uuid.uuid4().hex}.mp3"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        file.save(tmp.name)
        dest = f"reminder_audio/{fname}"
        public_url = upload_to_storage(tmp.name, dest, content_type="audio/mpeg")
        data["voiceUrl"] = public_url

    if "enabled" not in data:
        data["enabled"] = True
    if "days" not in data:
        data["days"] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    if "time" not in data and data.get("customTime"):
        data["time"] = data["customTime"]

    row = {
        "user_key": user_key,
        "payload": data,
        "created_at": time.time(),
    }

    try:
        res = supabase.table("legacy_reminders").insert(row).execute()
        inserted = (res.data or [None])[0]
        reminder_id = inserted["id"] if inserted else None
    except Exception as e:
        print("Supabase legacy_reminders insert error:", e)
        reminder_id = None

    return jsonify({"success": True, "id": reminder_id})


# ==================== VOICE UPLOAD (MP3 via FFMPEG) ====================

@app.route("/upload-voice", methods=["POST"])
def upload_voice():
    print("📥 /upload-voice: request received")
    file = request.files.get("file")
    if not file:
        print("⚠️ /upload-voice: no file sent")
        return jsonify({"success": False, "message": "no file"}), 400

    original_name = file.filename or "recording.m4a"
    base, ext = os.path.splitext(original_name)
    in_ext = (ext or ".m4a").lower()
    if in_ext not in [".m4a", ".mp4", ".aac", ".wav", ".mp3"]:
        in_ext = ".m4a"

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=in_ext)
    file.save(tmp_in.name)
    in_size = os.path.getsize(tmp_in.name)
    print(f"✅ Saved input audio as {tmp_in.name} (ext={in_ext}, size={in_size} bytes)")

    if in_size == 0:
        print("✗ Input file is empty")
        return jsonify({"success": False, "message": "uploaded audio is empty"}), 400

    if in_ext == ".mp3":
        print("ℹ️ Input is already MP3, uploading directly")
        out_path = tmp_in.name
        out_ext = ".mp3"
        content_type = "audio/mpeg"
    else:
        out_ext = ".mp3"
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=out_ext)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_in.name,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-b:a",
            "96k",
            tmp_out.name,
        ]
        print("🎛 Running ffmpeg:", " ".join(cmd))

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print("✗ ffmpeg conversion error:", e)
            stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
            print("ffmpeg stderr:\n", stderr)
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "ffmpeg conversion failed",
                        "stderr": stderr[:5000],
                    }
                ),
                500,
            )

        out_size = os.path.getsize(tmp_out.name)
        print(f"✅ ffmpeg OK, output={tmp_out.name}, size={out_size} bytes")

        if out_size == 0:
            print("✗ ffmpeg output is empty")
            return jsonify({"success": False, "message": "ffmpeg output is empty"}), 500

        out_path = tmp_out.name
        content_type = "audio/mpeg"

    fname = f"voices/voice-{uuid.uuid4().hex}{out_ext}"
    public_url = upload_to_storage(out_path, fname, content_type=content_type)

    print(f"✅ Uploaded voice to Supabase Storage: {public_url}")
    return jsonify({"success": True, "url": public_url})


# ==================== TTS SIMPLE ====================


@app.route("/synthesize", methods=["POST"])
def synthesize_endpoint():
    data = request.json or {}
    text = data.get("text", "")
    lang = data.get("lang", "en-IN")
    audio = synthesize_text(text, language_code=lang)
    fname = f"tts/tts-{uuid.uuid4().hex}.mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(audio)
    tmp.flush()
    public_url = upload_to_storage(tmp.name, fname, content_type="audio/mpeg")
    return jsonify({"success": True, "url": public_url})


# ==================== EMOTION PIPELINE ====================


@app.route("/process-emotion", methods=["POST"])
def process_emotion():
    try:
        file = request.files.get("file")
        form = request.form or {}
        lang = form.get("lang", "en")
        mobile = form.get("mobile") or form.get("userMobile") or ""
        primary = form.get("primary") or mobile
        chw = form.get("chw") or None
        text_override = form.get("text")

        if not file and not text_override:
            return jsonify({"success": False, "message": "no audio or text provided"}), 400

        user_key = normalize_user_key(mobile)

        audio_public_url = None
        audio_bytes = None

        if file:
            fname = f"mood_audio/mood-{uuid.uuid4().hex}.wav"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            file.save(tmp.name)

            with open(tmp.name, "rb") as f:
                audio_bytes = f.read()

            audio_public_url = upload_to_storage(tmp.name, fname, content_type="audio/wav")

        transcript = text_override or ""
        gcloud_lang = LANG_TO_GCLOUD.get(lang, "en-IN")

        if not transcript and audio_bytes:
            try:
                transcript = speech_to_text_bytes(audio_bytes, language_code=gcloud_lang)
                print(f"✓ STT Success - Transcript: {repr(transcript)}")
            except Exception as e:
                print(f"✗ STT error: {e}")
                transcript = ""

        if not transcript or transcript.strip() == "":
            print("⚠️ WARNING: Empty transcript detected!")
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Could not transcribe audio or empty text provided",
                        "transcript": transcript,
                        "sentiment": {"score": 0.0, "magnitude": 0.0},
                        "mood_label": "unknown",
                    }
                ),
                400,
            )

        sentiment = {"score": 0.0, "magnitude": 0.0}

        print(f"📝 Transcript raw: {repr(transcript)}, lang: {lang}")

        analysis_text = transcript
        analysis_lang = (lang or "en").lower()

        if analysis_lang != "en":
            try:
                translated = translate_text(transcript, target_lang="en", source_lang=analysis_lang)
                if translated:
                    print(f"🌐 Translated to EN: {translated}")
                    analysis_text = translated
                else:
                    print("⚠️ Translation returned empty, using original")
            except Exception as e:
                print(f"✗ Sentiment translate error: {e}")

        if not analysis_text or analysis_text.strip() == "":
            print("⚠️ WARNING: Empty analysis_text after translation!")
            analysis_text = transcript

        cloud_sentiment = {"score": 0.0, "magnitude": 0.0}
        try:
            cloud_sentiment = analyze_sentiment(analysis_text)
            print(f"☁️ Cloud NLP sentiment: {cloud_sentiment}")
        except Exception as e:
            print(f"✗ Sentiment error (Cloud NLP): {e}")

        heuristic = heuristic_sentiment(analysis_text)
        print(f"🔍 Heuristic sentiment: {heuristic}")

        try:
            cs = float(cloud_sentiment.get("score", 0.0))
            cm = float(cloud_sentiment.get("magnitude", 0.0))
        except Exception:
            cs, cm = 0.0, 0.0

        try:
            hs = float(heuristic.get("score", 0.0))
            hm = float(heuristic.get("magnitude", 0.0))
        except Exception:
            hs, hm = 0.0, 0.0

        cloud_strength = abs(cs) + cm
        heuristic_strength = abs(hs) + hm

        print(f"💪 Cloud strength: {cloud_strength}, Heuristic strength: {heuristic_strength}")

        if heuristic_strength >= cloud_strength or (cs == 0.0 and cm == 0.0 and hm > 0.0):
            print("✓ Using HEURISTIC sentiment")
            sentiment = heuristic
        else:
            print("✓ Using CLOUD sentiment")
            sentiment = cloud_sentiment

        score = float(sentiment.get("score", 0.0))
        magnitude = float(sentiment.get("magnitude", 0.0))
        print(f"🎯 Final sentiment: {sentiment}, Score: {score}, Magnitude: {magnitude}")

        if score <= -0.6:
            mood_label = "very_negative"
            escalate = True
        elif score <= -0.25:
            mood_label = "negative"
            escalate = True
        elif score < 0.25:
            mood_label = "neutral"
            escalate = False
        elif score < 0.6:
            mood_label = "positive"
            escalate = False
        else:
            mood_label = "very_positive"
            escalate = False

        print(f"😊 Mood label: {mood_label}, Escalate: {escalate}")

        if mood_label in ("very_negative", "negative"):
            base_en = (
                "I understand you are not feeling well. "
                "I am really sorry to hear that. Please take some rest and drink some water. "
            )
            if escalate:
                base_en += "I will now call your family member so they can check on you."
        elif mood_label == "neutral":
            base_en = (
                "Thank you for sharing how you feel. "
                "If you feel any discomfort or pain, please let me know, and I can call your family member."
            )
        else:
            base_en = (
                "I am happy to hear you are feeling okay. "
                "Please continue to take care of yourself and your medicines regularly."
            )

        dynamic_en = generate_dynamic_reply(transcript, mood_label, base_en)

        ai_text = dynamic_en
        if lang and lang != "en":
            try:
                ai_text = translate_text(dynamic_en, target_lang=lang)
            except Exception as e:
                print(f"✗ Translate error: {e}")
                ai_text = dynamic_en

        tts_public_url = None
        try:
            tts_audio = synthesize_text(ai_text, language_code=gcloud_lang)
            tts_name = f"tts_mood/tts-mood-{uuid.uuid4().hex}.mp3"
            tmp_tts = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp_tts.write(tts_audio)
            tmp_tts.flush()
            tts_public_url = upload_to_storage(tmp_tts.name, tts_name, content_type="audio/mpeg")
        except Exception as e:
            print(f"✗ TTS error: {e}")

        event_payload = {
            "mobile": mobile,
            "primary": primary,
            "chw": chw,
            "lang": lang,
            "transcript": transcript,
            "sentiment": sentiment,
            "mood_label": mood_label,
            "audio_url": audio_public_url,
            "tts_url": tts_public_url,
            "escalate": escalate,
            "ai_text": ai_text,
            "created_at": time.time(),
        }

        try:
            supabase.table("mood_events").insert(
                {
                    "user_key": user_key,
                    "created_at": event_payload["created_at"],
                    "payload": event_payload,
                }
            ).execute()
        except Exception as e:
            print("✗ mood_events insert error:", e)

        escalation_triggered = False
        if escalate and twilio_client and TWILIO_FROM and primary:
            escalation_triggered = True
            escalation_message = (
                "This is an automated call from the health companion for your parent or family member. "
                "They just reported feeling unwell. Please check on them as soon as possible."
            )

            try:
                attempts = 0
                for _ in range(3):
                    attempts += 1
                    try:
                        twilio_client.calls.create(
                            to=primary,
                            from_=TWILIO_FROM,
                            twiml=f"<Response><Say>{escalation_message}</Say></Response>",
                        )
                        break
                    except Exception as ce:
                        print(f"✗ Twilio primary call attempt {attempts} error: {ce}")

                if chw:
                    try:
                        chw_message = (
                            "This is an automated call for the community health worker. "
                            "The patient just reported feeling unwell and their family member "
                            "could not be reached. Please check on them as soon as possible."
                        )
                        twilio_client.calls.create(
                            to=chw,
                            from_=TWILIO_FROM,
                            twiml=f"<Response><Say>{chw_message}</Say></Response>",
                        )
                    except Exception as ce:
                        print(f"✗ Twilio CHW call error: {ce}")
            except Exception as e:
                print(f"✗ Twilio escalation error: {e}")

        return jsonify(
            {
                "success": True,
                "transcript": transcript,
                "sentiment": sentiment,
                "mood_label": mood_label,
                "escalate": escalate,
                "tts_url": tts_public_url,
                "audio_url": audio_public_url,
                "escalation_triggered": escalation_triggered,
                "ai_text": ai_text,
            }
        )

    except Exception as e:
        print(f"✗ process_emotion error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": "internal error", "error": str(e)}), 500


# ==================== MOOD HISTORY ====================


@app.route("/mood-history", methods=["GET"])
def mood_history():
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = (
            supabase.table("mood_events")
            .select("id, created_at, payload")
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        items = []
        for row in rows:
            val = dict(row.get("payload") or {})
            val["id"] = row["id"]
            items.append(val)

        items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        items = items[:10]

        return jsonify({"success": True, "items": items})
    except Exception as e:
        print(f"✗ mood_history error: {e}")
        return jsonify({"success": False, "message": "error", "error": str(e)}), 500


@app.route("/me/config", methods=["GET"])
def me_config():
    return jsonify({"notification": None, "reminder": None})


# ==================== FAMILY MEMBERS ====================


@app.route("/family-members", methods=["GET"])
def get_family_members():
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = (
            supabase.table("family_members")
            .select("id, created_at, payload")
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        members = []
        for row in rows:
            v = dict(row.get("payload") or {})
            v["id"] = row["id"]
            members.append(v)

        members.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jsonify({"success": True, "members": members})
    except Exception as e:
        print(f"✗ get_family_members error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/family-members", methods=["POST"])
def add_family_member():
    data = request.json or {}
    mobile = data.get("user_mobile") or ""
    user_key = normalize_user_key(mobile)

    member_data = {
        "name": data.get("name"),
        "mobile": data.get("mobile"),
        "relationship": data.get("relationship"),
        "age": data.get("age"),
        "notes": data.get("notes", ""),
        "created_at": time.time(),
    }

    try:
        row = {
            "user_key": user_key,
            "created_at": member_data["created_at"],
            "payload": member_data,
        }
        res = supabase.table("family_members").insert(row).execute()
        inserted = (res.data or [None])[0]
        if inserted:
            member_data["id"] = inserted["id"]
            return jsonify({"success": True, "id": inserted["id"], "member": member_data})
        else:
            return jsonify({"success": False, "error": "insert failed"}), 500
    except Exception as e:
        print(f"✗ add_family_member error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/family-members/<member_id>", methods=["PUT"])
def update_family_member(member_id):
    data = request.json or {}
    mobile = data.get("user_mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = (
            supabase.table("family_members")
            .select("payload")
            .eq("id", member_id)
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify({"success": False, "error": "member not found"}), 404

        payload = dict(rows[0]["payload"] or {})
        payload.update(data)

        supabase.table("family_members").update({"payload": payload}).eq("id", member_id).eq(
            "user_key", user_key
        ).execute()
        return jsonify({"success": True})
    except Exception as e:
        print(f"✗ update_family_member error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/family-members/<member_id>", methods=["DELETE"])
def delete_family_member(member_id):
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        supabase.table("family_members").delete().eq("id", member_id).eq("user_key", user_key).execute()
        return jsonify({"success": True})
    except Exception as e:
        print(f"✗ delete_family_member error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== VOICE RECORDINGS ====================


@app.route("/voice-recordings", methods=["GET"])
def get_voice_recordings():
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = (
            supabase.table("voice_recordings")
            .select("id, created_at, payload")
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        recordings = []
        for row in rows:
            v = dict(row.get("payload") or {})
            v["id"] = row["id"]
            recordings.append(v)

        recordings.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jsonify({"success": True, "recordings": recordings})
    except Exception as e:
        print(f"✗ get_voice_recordings error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/voice-recordings", methods=["POST"])
def add_voice_recording_route():
    file = request.files.get("file")
    form = request.form or {}

    mobile = form.get("user_mobile") or ""
    user_key = normalize_user_key(mobile)

    if not file:
        return jsonify({"success": False, "message": "no file"}), 400

    # ---- 1) Save input temp file with original extension ----
    original_name = file.filename or "recording.m4a"
    base, ext = os.path.splitext(original_name)
    in_ext = (ext or ".m4a").lower()
    if in_ext not in [".m4a", ".mp4", ".aac", ".wav", ".mp3", ".caf"]:
        in_ext = ".m4a"

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=in_ext)
    file.save(tmp_in.name)
    in_size = os.path.getsize(tmp_in.name)
    print(f"✅ [voice-recordings] Saved input {tmp_in.name} (ext={in_ext}, size={in_size} bytes)")

    if in_size == 0:
        return jsonify({"success": False, "message": "uploaded audio is empty"}), 400

    # ---- 2) Convert to MP3 (unless already MP3) ----
    if in_ext == ".mp3":
        print("ℹ️ [voice-recordings] Input already MP3, using as-is")
        out_path = tmp_in.name
    else:
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", tmp_in.name,
            "-ac", "1",
            "-ar", "44100",
            "-b:a", "96k",
            tmp_out.name,
        ]
        print("[voice-recordings] 🎛 Running ffmpeg:", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
            print("✗ [voice-recordings] ffmpeg error:", stderr)
            return jsonify({
                "success": False,
                "message": "ffmpeg conversion failed",
                "stderr": stderr[:5000],
            }), 500

        out_size = os.path.getsize(tmp_out.name)
        print(f"✅ [voice-recordings] ffmpeg OK, output={tmp_out.name}, size={out_size} bytes")
        if out_size == 0:
            return jsonify({"success": False, "message": "ffmpeg output is empty"}), 500

        out_path = tmp_out.name

    # ---- 3) Upload MP3 to storage using your helper (SUPABASE_STORAGE_BUCKET inside) ----
    storage_key = f"voice_recordings/{user_key}/voice-recording-{uuid.uuid4().hex}.mp3"
    public_url = upload_to_storage(out_path, storage_key, content_type="audio/mpeg")
    print(f"✅ [voice-recordings] Stored MP3 at {public_url}")

    # ---- 4) Transcribe from the MP3 (optional) ----
    transcript = ""
    try:
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        lang = form.get("language", "en")
        gcloud_lang = LANG_TO_GCLOUD.get(lang, "en-IN")
        transcript = speech_to_text_bytes(audio_bytes, language_code=gcloud_lang)
    except Exception as e:
        print(f"⚠️ [voice-recordings] Transcription failed: {e}")

    # ---- 5) Prepare metadata + insert into Supabase table ----
    now_ts = time.time()
    recording_data = {
        "label": form.get("label", "Untitled Recording"),
        "speaker_name": form.get("speaker_name", ""),
        "speaker_relationship": form.get("speaker_relationship", ""),
        "audio_url": public_url,  # ✅ MP3 URL
        "duration_seconds": int(form.get("duration", 0)),
        "transcript": transcript,
        "language": form.get("language", "en"),
        "created_at": now_ts,
    }

    try:
        row = {
            "user_key": user_key,
            "created_at": now_ts,
            "payload": recording_data,
        }
        res = supabase.table("voice_recordings").insert(row).execute()
        inserted = (res.data or [None])[0]
        if inserted:
            recording_data["id"] = inserted["id"]
            return jsonify({"success": True, "recording": recording_data})
        else:
            return jsonify({"success": False, "error": "insert failed"}), 500
    except Exception as e:
        print(f"✗ add_voice_recording error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/voice-recordings/<recording_id>", methods=["DELETE"])
def delete_voice_recording(recording_id):
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        supabase.table("voice_recordings").delete().eq("id", recording_id).eq("user_key", user_key).execute()
        return jsonify({"success": True})
    except Exception as e:
        print(f"✗ delete_voice_recording error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== MEDICATION REMINDERS (USED BY SCHEDULER) ====================


@app.route("/medication-reminders", methods=["GET"])
def get_medication_reminders():
    """
    Fetch all medication reminders for a given mobile from Supabase 'reminders' table.
    """
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = supabase.table("reminders").select("*").eq("user_key", user_key).execute()

        if getattr(res, "error", None):
            print("✗ get_medication_reminders Supabase error:", res.error)
            return (
                jsonify({"success": False, "error": str(res.error)}),
                500,
            )

        reminders: list[dict] = []
        for row in res.data or []:
            payload = row.get("payload") or {}
            item = dict(payload)
            item["id"] = row.get("id")
            reminders.append(item)

        # Optional: sort by time for display
        def sort_key(r):
            # "HH:MM" or fallback
            return str(r.get("time") or "")

        reminders.sort(key=sort_key)

        return jsonify({"success": True, "reminders": reminders})

    except Exception as e:
        print("✗ get_medication_reminders error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/medication-reminders", methods=["POST"])
def add_medication_reminder():
    """
    Create a new medication reminder in Supabase 'reminders' table.

    Table structure (recommended):
      - id         (uuid, primary key, default uuid_generate_v4())
      - user_key   (text)
      - created_at (double precision or timestamptz)
      - payload    (jsonb)
    """
    data = request.json or {}

    mobile = (
        data.get("user_mobile")
        or data.get("mobile")
        or ""
    )
    user_key = normalize_user_key(mobile)

    member_mobile = (
        data.get("member_mobile")
        or data.get("familyMemberMobile")
        or data.get("family_member_mobile")
        or ""
    )

    # Time string "HH:MM"
    raw_time = (
        data.get("time")
        or data.get("customTime")
        or ""
    )

    time_str = normalize_time_str(raw_time)
    if not time_str:
        # Don't save broken data – tell frontend it's invalid
        return jsonify({
            "success": False,
            "error": f"Invalid reminder time: {repr(raw_time)}. Expected HH:MM, e.g. '01:38'."
        }), 400

    # Voice URL (must be MP3 — we already fixed /voice-recordings to return MP3 URLs)
    voice_url = (
        data.get("voice_url")
        or data.get("voiceUrl")
        or data.get("audio_url")
    )

    voice_label = (
        data.get("voice_label")
        or data.get("voiceLabel")
        or None
    )

    all_days = data.get("days") or ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    enabled = True if data.get("enabled") is None else bool(data.get("enabled"))

    now_ts = time.time()

    # This is what the frontend will expect back
    reminder_payload = {
        "user_mobile": mobile,
        "member_id": data.get("member_id"),
        "member_name": data.get("member_name"),
        "member_mobile": member_mobile,
        "medication_name": data.get("medication_name") or data.get("medicationName"),
        "time": time_str,
        "days": all_days,
        "voice_url": voice_url,
        "voice_label": voice_label,
        "enabled": enabled,
        "fallback_chw": data.get("fallback_chw", ""),
        "created_at": now_ts,
    }

    try:
        row = {
            "user_key": user_key,
            "created_at": now_ts,
            "payload": reminder_payload,
        }

        res = supabase.table("reminders").insert(row).execute()

        if getattr(res, "error", None):
            print("✗ add_medication_reminder Supabase error:", res.error)
            return (
                jsonify({"success": False, "error": str(res.error)}),
                500,
            )

        inserted = (res.data or [None])[0]
        if not inserted:
            return jsonify({"success": False, "error": "insert failed"}), 500

        # We DO NOT assume inserted["created_at"] exists.
        reminder_payload["id"] = inserted.get("id")

        return jsonify(
            {
                "success": True,
                "id": reminder_payload["id"],
                "reminder": reminder_payload,
            }
        )

    except Exception as e:
        print("✗ add_medication_reminder error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/medication-reminders/<reminder_id>", methods=["PUT"])
def update_medication_reminder(reminder_id):
    data = request.json or {}
    mobile = data.get("user_mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = (
            supabase.table("reminders")
            .select("payload")
            .eq("id", reminder_id)
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify({"success": False, "error": "reminder not found"}), 404

        payload = dict(rows[0]["payload"] or {})
        payload.update(data)

        update_fields = {"payload": payload}
        if "time" in payload:
            update_fields["time"] = payload["time"]
        if "enabled" in payload:
            update_fields["enabled"] = payload["enabled"]

        supabase.table("reminders").update(update_fields).eq("id", reminder_id).eq(
            "user_key", user_key
        ).execute()
        return jsonify({"success": True})
    except Exception as e:
        print(f"✗ update_medication_reminder error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/medication-reminders/<reminder_id>/toggle", methods=["POST"])
def toggle_medication_reminder(reminder_id):
    data = request.json or {}
    mobile = data.get("user_mobile") or ""
    user_key = normalize_user_key(mobile)
    enabled = data.get("enabled", True)

    try:
        res = (
            supabase.table("reminders")
            .select("payload")
            .eq("id", reminder_id)
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return jsonify({"success": False, "error": "reminder not found"}), 404

        payload = dict(rows[0]["payload"] or {})
        payload["enabled"] = bool(enabled)

        supabase.table("reminders").update(
            {"enabled": bool(enabled), "payload": payload}
        ).eq("id", reminder_id).eq("user_key", user_key).execute()

        return jsonify({"success": True, "enabled": bool(enabled)})
    except Exception as e:
        print(f"✗ toggle_medication_reminder error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/medication-reminders/<reminder_id>", methods=["DELETE"])
def delete_medication_reminder(reminder_id):
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        supabase.table("reminders").delete().eq("id", reminder_id).eq("user_key", user_key).execute()
        return jsonify({"success": True})
    except Exception as e:
        print(f"✗ delete_medication_reminder error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== AUDIO PROXY FOR TWILIO ====================


@app.route("/proxy-audio", methods=["GET"])
def proxy_audio():
    src = request.args.get("url")
    if not src:
        return "Missing 'url' query parameter", 400

    try:
        r = requests.get(src, stream=True)
        if r.status_code != 200:
            print("⚠️ proxy-audio: source returned", r.status_code)
            return "Failed to fetch source audio", 502

        return Response(r.content, mimetype="audio/mpeg")
    except Exception as e:
        print("✗ proxy-audio error:", e)
        return "Internal proxy error", 500


# ==================== TWILIO TEST CALL ====================


@app.route("/call-reminder-test", methods=["POST"])
def call_reminder_test():
    try:
        if not twilio_client or not TWILIO_FROM:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Twilio is not configured on the server.",
                    }
                ),
                500,
            )

        data = request.get_json() or {}
        to = data.get("to")
        voice_url = (
            data.get("voiceUrl")
            or data.get("voice_url")
            or data.get("audio_url")
        )

        if not to or not voice_url:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "Both 'to' and 'voice_url' are required.",
                    }
                ),
                400,
            )

        safe_url = escape(voice_url)

        twiml = f"""
<Response>
    <Play>{safe_url}</Play>
</Response>
        """.strip()

        print("📞 Twilio call_reminder_test TwiML:\n", twiml)

        call = twilio_client.calls.create(
            to=to,
            from_=TWILIO_FROM,
            twiml=twiml,
        )

        return jsonify(
            {"success": True, "message": "Test call started.", "call_sid": call.sid}
        )
    except Exception as e:
        print("call_reminder_test error:", e)
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Failed to start test call.",
                    "error": str(e),
                }
            ),
            500,
        )


# ==================== TRIGGER REMINDER CALL (USED BY SCHEDULER) ====================

from urllib.parse import quote_plus
from xml.sax.saxutils import escape

@app.route('/trigger-reminder-call', methods=['POST'])
def trigger_reminder_call_endpoint():
    """
    Trigger a medication reminder call that runs the interactive Twilio med-flow.

    Expects JSON:
      { "reminder_id": "<id>", "user_mobile": "<user's login mobile>" }
    """
    data = request.json or {}
    reminder_id = data.get('reminder_id')
    user_mobile = data.get('user_mobile')

    if not reminder_id or not user_mobile:
        return jsonify({
            "success": False,
            "message": "Both 'reminder_id' and 'user_mobile' are required.",
        }), 400

    user_key = normalize_user_key(user_mobile)

    # Load reminder from Supabase 'reminders' table
    try:
        res = supabase.table("reminders").select("*").eq("id", reminder_id).execute()
        rows = res.data or []
        if not rows:
            return jsonify({"success": False, "message": "Reminder not found"}), 404

        row = rows[0]
        payload = row.get("payload") or {}

        if not payload.get("enabled", True):
            return jsonify({"success": False, "message": "Reminder is disabled"}), 400

        # Destination to call
        member_mobile = (
            payload.get("member_mobile")
            or payload.get("familyMemberMobile")
            or payload.get("family_member_mobile")
        )
        if not member_mobile:
            return jsonify({
                "success": False,
                "message": "No destination number found on reminder.",
            }), 400

        if not twilio_client or not TWILIO_FROM:
            return jsonify({"success": False, "message": "Twilio not configured"}), 500

        # Start Twilio call pointing to /twilio/med-flow (so Q&A will run)
        base_url = (os.environ.get("PUBLIC_BASE_URL") or request.host_url).rstrip("/")
        start_url = (
            f"{base_url}/twilio/med-flow"
            f"?reminder_id={quote_plus(reminder_id)}"
            f"&user_key={quote_plus(user_key)}"
            f"&q_index=0"
        )

        print("📞 Twilio outbound med-flow call URL:", start_url)

        call = twilio_client.calls.create(
            to=member_mobile,
            from_=TWILIO_FROM,
            url=start_url,   # 👈 Twilio will fetch TwiML from /twilio/med-flow
        )

        # Optional: log basic call info in Supabase `call_logs`
        try:
            log_payload = {
                "reminder_id": reminder_id,
                "member_mobile": member_mobile,
                "scheduled_time": payload.get("time"),
                "actual_call_time": time.time(),
                "call_status": "initiated",
                "twilio_call_sid": call.sid,
            }
            supabase.table("call_logs").insert({
                "user_key": user_key,
                "created_at": time.time(),
                "payload": log_payload,
            }).execute()
        except Exception as e:
            print("⚠️ call_logs insert error:", e)

        return jsonify({
            "success": True,
            "call_sid": call.sid,
            "message": "Reminder call started with med-flow.",
        })

    except Exception as e:
        print("✗ trigger_reminder_call error:", e)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/twiml-play-audio", methods=["GET", "POST"])
def twiml_play_audio():
    audio_url = request.args.get("url") or request.form.get("url")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
</Response>"""

    return twiml, 200, {"Content-Type": "text/xml"}


# ==================== CALL LOGS ====================


@app.route("/call-logs", methods=["GET"])
def get_call_logs():
    mobile = request.args.get("mobile") or ""
    user_key = normalize_user_key(mobile)

    try:
        res = (
            supabase.table("call_logs")
            .select("id, created_at, payload")
            .eq("user_key", user_key)
            .execute()
        )
        rows = res.data or []
        logs = []
        for row in rows:
            v = dict(row.get("payload") or {})
            v["id"] = row["id"]
            logs.append(v)

        logs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        logs = logs[:20]

        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        print(f"✗ get_call_logs error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== SCHEDULER DEBUG ENDPOINTS ====================


@app.route("/test-scheduler", methods=["GET"])
def test_scheduler_endpoint():
    test_scheduler()
    return jsonify({"success": True, "message": "Scheduler test triggered"})


@app.route("/scheduler-status", methods=["GET"])
def scheduler_status():
    from scheduler_service import scheduler

    return jsonify(
        {
            "success": True,
            "running": scheduler.running,
            "jobs": [
                {
                    "id": job.id,
                    "next_run": str(job.next_run_time) if job.next_run_time else None,
                }
                for job in scheduler.get_jobs()
            ],
        }
    )


# ==================== TWILIO MEDICATION FLOW ====================

@app.route("/twilio/med-flow", methods=["GET", "POST"])
def twilio_med_flow():
    """
    Step handler for medication Q&A flow.

    Query params:
      - reminder_id
      - user_key
      - q_index (0-based index of which question to ask)

    We load reminder payload from Supabase and:
      - If there are questions and q_index < len(questions): ask that question and Gather speech.
      - Else: say goodbye and hang up.
    """
    reminder_id = request.args.get("reminder_id")
    user_key = request.args.get("user_key") or "anon"
    q_index_raw = request.args.get("q_index", "0")

    try:
        q_index = int(q_index_raw)
    except ValueError:
        q_index = 0

    # Load reminder from Supabase
    try:
        res = supabase.table("reminders").select("*").eq("id", reminder_id).execute()
        rows = res.data or []
        if not rows:
            vr = VoiceResponse()
            vr.say("Sorry, we could not find your reminder.", voice="alice", language="en-IN")
            vr.hangup()
            return Response(str(vr), mimetype="text/xml")

        row = rows[0]
        payload = row.get("payload") or {}
    except Exception as e:
        print("✗ med-flow Supabase load error:", e)
        vr = VoiceResponse()
        vr.say("There was an error loading the reminder. Goodbye.", voice="alice", language="en-IN")
        vr.hangup()
        return Response(str(vr), mimetype="text/xml")

    # Questions configuration
    questions = payload.get("call_questions") or []

    # If no custom questions defined, use a default 2-step flow
    if not questions:
        questions = [
            {
                "id": "taken_meds",
                "text": "Have you taken your tablets?",
                "audio_url": None,
            },
            {
                "id": "feeling_now",
                "text": "How are you feeling now?",
                "audio_url": None,
            },
        ]

    vr = VoiceResponse()

    # If no more questions -> goodbye
    if q_index >= len(questions):
        goodbye_audio = payload.get("goodbye_audio_url")
        if goodbye_audio:
            vr.play(goodbye_audio)
        else:
            vr.say(
                "Thank you for your time. Please continue to take care and remember your medicines.",
                voice="alice",
                language="en-IN",
            )
        vr.hangup()
        return Response(str(vr), mimetype="text/xml")

    question = questions[q_index]
    question_id = question.get("id", f"q{q_index+1}")
    question_audio = question.get("audio_url")
    question_text = question.get("text") or "How are you?"

    base_url = (os.environ.get("PUBLIC_BASE_URL") or request.host_url).rstrip("/")
    answer_url = (
        f"{base_url}/twilio/med-flow-answer"
        f"?reminder_id={quote_plus(reminder_id)}"
        f"&user_key={quote_plus(user_key)}"
        f"&q_index={q_index}"
        f"&question_id={quote_plus(question_id)}"
    )

    # Optional: play the reminder voice first on the very first step
    if q_index == 0:
        reminder_voice = (
            payload.get("voice_url")
            or payload.get("voiceUrl")
            or payload.get("audio_url")
        )
        if reminder_voice:
            vr.play(reminder_voice)

    # Ask question (audio if present, otherwise TTS)
    if question_audio:
        vr.play(question_audio)
    else:
        vr.say(question_text, voice="alice", language="en-IN")

    g = vr.gather(
        input="speech",
        action=answer_url,
        method="POST",
        timeout=7,
    )
    if not question_audio:
        g.say("Please answer after the tone.", voice="alice", language="en-IN")

    # In case no input, redirect back to same question (or next step as you prefer)
    vr.redirect(answer_url)

    return Response(str(vr), mimetype="text/xml")

@app.route("/twilio/med-flow-answer", methods=["POST"])
def twilio_med_flow_answer():
    """
    Receive an answer from Twilio (speech result) for one question,
    store it in Supabase (medication_call_sessions), then redirect
    back to /twilio/med-flow for the next question.
    """
    reminder_id = request.args.get("reminder_id")
    user_key = request.args.get("user_key") or "anon"
    question_id = request.args.get("question_id") or "q"
    q_index_raw = request.args.get("q_index", "0")

    try:
        q_index = int(q_index_raw)
    except ValueError:
        q_index = 0

    call_sid = request.form.get("CallSid")
    from_number = request.form.get("From")
    speech_result = request.form.get("SpeechResult") or ""
    confidence = request.form.get("Confidence")

    print("📝 med-flow-answer:", {
        "call_sid": call_sid,
        "from": from_number,
        "question_id": question_id,
        "q_index": q_index,
        "speech": speech_result,
        "confidence": confidence,
    })

    # Load reminder payload (for member mobile etc. if needed)
    try:
        res = supabase.table("reminders").select("*").eq("id", reminder_id).execute()
        rows = res.data or []
        payload = (rows[0].get("payload") if rows else {}) or {}
    except Exception as e:
        print("✗ med-flow-answer load reminder error:", e)
        payload = {}

    # Upsert session row in Supabase
    try:
        # 1) Check if session already exists for this call_sid + user_key
        sess_res = supabase.table("medication_call_sessions") \
            .select("*") \
            .eq("user_key", user_key) \
            .eq("call_sid", call_sid) \
            .execute()

        sessions = sess_res.data or []
        now_ts = time.time()

        answer_entry = {
            "question_id": question_id,
            "q_index": q_index,
            "answer_text": speech_result,
            "from": from_number,
            "confidence": confidence,
            "received_at": now_ts,
        }

        if not sessions:
            # Create new session
            session_payload = {
                "reminder_id": reminder_id,
                "member_mobile": payload.get("member_mobile"),
                "answers": [answer_entry],
            }
            supabase.table("medication_call_sessions").insert({
                "user_key": user_key,
                "call_sid": call_sid,
                "started_at": now_ts,
                "last_updated_at": now_ts,
                "payload": session_payload,
            }).execute()
        else:
            # Update existing session: append answer to answers list
            sess_row = sessions[0]
            sess_payload = sess_row.get("payload") or {}
            answers = sess_payload.get("answers") or []
            answers.append(answer_entry)
            sess_payload["answers"] = answers

            supabase.table("medication_call_sessions").update({
                "last_updated_at": now_ts,
                "payload": sess_payload,
            }).eq("id", sess_row["id"]).execute()

    except Exception as e:
        print("✗ Supabase update in med-flow-answer:", e)

    # Redirect to next question
    next_q_index = q_index + 1
    base_url = (os.environ.get("PUBLIC_BASE_URL") or request.host_url).rstrip("/")
    next_url = (
        f"{base_url}/twilio/med-flow"
        f"?reminder_id={quote_plus(reminder_id)}"
        f"&user_key={quote_plus(user_key)}"
        f"&q_index={next_q_index}"
    )

    vr = VoiceResponse()
    vr.redirect(next_url)
    return Response(str(vr), mimetype="text/xml")

# ==================== REMINDERS DEBUG ====================


@app.route("/debug/reminders", methods=["GET"])
def debug_reminders():
    try:
        res = supabase.table("reminders").select("id, user_key, payload").execute()
        rows = res.data or []

        flat = []
        for row in rows:
            v = dict(row.get("payload") or {})
            v["id"] = row["id"]
            v["user_key"] = row["user_key"]
            flat.append(v)

        user_keys = sorted({r["user_key"] for r in flat})
        print(f"[debug/reminders] Total reminders: {len(flat)}, users: {user_keys}")

        return jsonify(
            {
                "success": True,
                "top_level": user_keys,
                "flat_reminders": flat,
                "flat_count": len(flat),
            }
        )
    except Exception as e:
        print("✗ debug_reminders error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== MAIN ====================


if __name__ == "__main__":
    host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_RUN_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)