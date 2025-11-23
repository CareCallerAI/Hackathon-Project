# google_client.py
"""
Google Cloud helper wrapper for:
- Text-to-Speech
- Speech-to-Text
- Translation (v3)
- Natural Language (sentiment, entities)

Requirements (pip):
  pip install google-cloud-texttospeech google-cloud-speech google-cloud-translate google-cloud-language google-auth

Set environment variable:
  GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/your-service-account.json
"""

import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from typing import Optional, List, Dict, Any

from google.oauth2 import service_account
from google.cloud import texttospeech_v1 as texttospeech
from google.cloud import speech_v1 as speech
from google.cloud import translate_v3 as translate_v3
from google.cloud import language_v1 as language_v1


# -------------------------
# Credentials & helpers
# -------------------------
def _load_credentials() -> service_account.Credentials:
    """Load service account credentials from GOOGLE_APPLICATION_CREDENTIALS env var."""
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path:
        raise EnvironmentError(
            "GOOGLE_APPLICATION_CREDENTIALS env var is not set. "
            "Set it to the service account JSON path."
        )

    cred_path = Path(cred_path)
    if not cred_path.exists():
        raise FileNotFoundError(f"Credentials file not found at: {cred_path}")

    creds = service_account.Credentials.from_service_account_file(str(cred_path))
    return creds


def _get_project_id(creds: service_account.Credentials) -> str:
    """Return project id from credentials."""
    if hasattr(creds, "project_id") and creds.project_id:
        return creds.project_id
    raise RuntimeError("Project ID not available in credentials.")


# -------------------------
# Text-to-Speech
# -------------------------
def synthesize_text(
    text: str,
    language_code: str = "en-IN",
    voice_name: Optional[str] = None,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
) -> bytes:
    """
    Synthesize text into audio (MP3) and return bytes.

    Args:
      text: input plain text (no SSML).
      language_code: e.g. 'en-IN', 'hi-IN', 'te-IN', 'bn-IN', etc.
      voice_name: specific voice name (optional). If None, API chooses a default.
      speaking_rate: multiplier for speaking rate (1.0 is default)
      pitch: pitch adjustment in semitones.

    Returns:
      bytes of MP3 audio.
    """
    creds = _load_credentials()
    client = texttospeech.TextToSpeechClient(credentials=creds)

    input_text = texttospeech.SynthesisInput(text=text)

    # If user provided voice_name use it, else request a neutral voice for language.
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name if voice_name else None,
        ssml_gender=texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate,
        pitch=pitch,
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    return response.audio_content


# -------------------------
# Speech-to-Text (GCS URI)
# -------------------------
def transcribe_gcs(
    gcs_uri: str,
    language_code: str = "en-IN",
    enable_automatic_punctuation: bool = True,
    timeout: int = 300,
) -> str:
    """
    Transcribe audio located at a Google Cloud Storage URI using long_running_recognize.

    Args:
      gcs_uri: e.g. 'gs://your-bucket/path/to/audio.mp3'
      language_code: language code like 'en-IN'
      timeout: seconds to wait for long running operation

    Returns:
      Transcribed text (concatenated).
    """
    creds = _load_credentials()
    client = speech.SpeechClient(credentials=creds)

    # ✅ For GCS, use URI. Do NOT use audio_bytes here.
    audio = speech.RecognitionAudio(uri=gcs_uri)

    # ✅ Let Google auto-detect encoding – no encoding field
    config = speech.RecognitionConfig(
        language_code=language_code,
        enable_automatic_punctuation=enable_automatic_punctuation,
    )

    operation = client.long_running_recognize(
        request={"config": config, "audio": audio}
    )

    response = operation.result(timeout=timeout)
    transcripts: List[str] = []
    for result in response.results:
        transcripts.append(result.alternatives[0].transcript)

    return " ".join(transcripts).strip()


def speech_to_text_bytes(
    audio_bytes: bytes,
    language_code: str = "en-IN",
    enable_automatic_punctuation: bool = True,
    sample_rate_hertz: int = 16000,
) -> str:
    """
    Speech-to-Text for raw audio bytes.
    Assumes LINEAR16 PCM at sample_rate_hertz (default 16 kHz).
    """
    creds = _load_credentials()
    client = speech.SpeechClient(credentials=creds)

    audio = speech.RecognitionAudio(content=audio_bytes)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # ✅ important
        sample_rate_hertz=sample_rate_hertz,                       # ✅ important
        language_code=language_code,
        enable_automatic_punctuation=enable_automatic_punctuation,
    )

    response = client.recognize(request={"config": config, "audio": audio})

    if not response.results:
        return ""

    texts: List[str] = []
    for result in response.results:
        texts.append(result.alternatives[0].transcript)
    return " ".join(texts).strip()


# -------------------------
# Translation (v3)
# -------------------------
def translate_text(
    text: str,
    target_lang: str = "en",
    source_lang: Optional[str] = None,
    location: str = "global",
) -> str:
    """
    Translate text using TranslationServiceClient (v3).

    Args:
      text: source text
      target_lang: 'en', 'hi', 'te', etc.
      source_lang: optional source language (e.g. 'hi'). If None, service will try detect.
      location: 'global' or a region like 'us-central1'

    Returns:
      translated text (string)
    """
    creds = _load_credentials()
    project_id = _get_project_id(creds)
    client = translate_v3.TranslationServiceClient(credentials=creds)

    parent = f"projects/{project_id}/locations/{location}"

    contents = [text]
    mime_type = "text/plain"

    request = {
        "parent": parent,
        "contents": contents,
        "mime_type": mime_type,
        "target_language_code": target_lang,
    }
    if source_lang:
        request["source_language_code"] = source_lang

    response = client.translate_text(request=request)
    if not response.translations:
        return ""
    return response.translations[0].translated_text


def detect_language(text: str, location: str = "global") -> Dict[str, Any]:
    """
    Detect language of given text. Returns the raw detection response details.
    """
    creds = _load_credentials()
    project_id = _get_project_id(creds)
    client = translate_v3.TranslationServiceClient(credentials=creds)
    parent = f"projects/{project_id}/locations/{location}"
    response = client.detect_language(
        request={
            "parent": parent,
            "content": text,
            "mime_type": "text/plain",
        }
    )
    # response.languages is a sequence of detected languages
    languages = [
        {"language_code": l.language_code, "confidence": l.confidence}
        for l in getattr(response, "languages", [])
    ]
    return {"languages": languages}


# -------------------------
# Natural Language (NLP)
# -------------------------
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using Cloud Natural Language API.
    Returns {'score': float, 'magnitude': float}
    """
    creds = _load_credentials()
    client = language_v1.LanguageServiceClient(credentials=creds)
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    result = client.analyze_sentiment(request={"document": document})
    sentiment = result.document_sentiment
    return {"score": sentiment.score, "magnitude": sentiment.magnitude}


def analyze_entities(text: str) -> List[Dict[str, Any]]:
    """
    Analyze entities in text (returns list of entities with name, type, salience, metadata).
    """
    creds = _load_credentials()
    client = language_v1.LanguageServiceClient(credentials=creds)
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
    resp = client.analyze_entities(request={"document": document})
    out: List[Dict[str, Any]] = []
    for ent in resp.entities:
        out.append(
            {
                "name": ent.name,
                "type": language_v1.Entity.Type(ent.type_).name if ent.type_ is not None else None,
                "salience": ent.salience,
                "wikipedia_url": ent.metadata.get("wikipedia_url"),
                "mid": ent.metadata.get("mid"),
            }
        )
    return out


# -------------------------
# Simple smoke test helpers (commented)
# -------------------------
if __name__ == "__main__":
    # Quick manual test (requires GOOGLE_APPLICATION_CREDENTIALS set)
    print("Google client smoke test")
    creds = _load_credentials()
    print("Project ID:", _get_project_id(creds))

    # Synthesize sample TTS
    try:
        b = synthesize_text("Hello from Google Cloud TTS", language_code="en-IN")
        with open("tts_sample.mp3", "wb") as f:
            f.write(b)
        print("Wrote tts_sample.mp3")
    except Exception as e:
        print("TTS error:", e)

    # Translation sample
    try:
        out = translate_text("नमस्ते", target_lang="en")
        print("Translate:", out)
    except Exception as e:
        print("Translate error:", e)

    # NLP sample
    try:
        s = analyze_sentiment("I am very happy")
        print("Sentiment:", s)
    except Exception as e:
        print("NLP error:", e)