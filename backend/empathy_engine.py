# empathy_engine.py (FINAL CODE: Multi-Turn, Gemini Priority, Multilingual Fallback)

from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os
from os import path # Required for file existence check
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

print("DEBUG: STARTING EMAPTHY ENGINE FILE EXECUTION...")

# Load environment variables first
load_dotenv()

# --- ElevenLabs Setup ---
try:
    ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_KEY:
        print("CRITICAL: ELEVENLABS_API_KEY not found.")
        eleven_client = None
    else:
        MOCK_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "50YSQEDPA2vlOxhCseP4")
        eleven_client = ElevenLabs(api_key=ELEVENLABS_KEY)
        print("DEBUG: ElevenLabs Client initialized successfully.")
except Exception as e:
    print(f"WARNING: ElevenLabs client initialization failed: {e}")
    eleven_client = None


# --- Gemini Client Setup (PRIORITY) ---
try:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        print("CRITICAL: GEMINI_API_KEY not found in .env. LLM will fail.")
        gemini_client = None
    else:
        # FIX: Explicitly pass the API key to the client for direct authentication
        gemini_client = genai.Client(api_key=GEMINI_KEY)

except Exception as e:
    print(f"CRITICAL: Gemini client failed to initialize: {e}")
    gemini_client = None


# --- MULTILINGUAL HARDCODED FALLBACK MAP ---
# FIX: Placed the map here so all functions below can see it.
FALLBACK_MAP = {
    'en-IN': "I'm having trouble with my connection right now. Can you please confirm again: did you take your medicine?",
    'hi-IN': "क्षमा करें, नेटवर्क में दिक्कत है. क्या आप एक बार और बता सकते हैं कि आपने दवाई ले ली है?",
    'te-IN': "క్షమించండి, నెట్‌వర్క్ సమస్య ఉంది. దయచేసి, మీరు మందు వేశారా అని మళ్లీ ఒకసారి చెప్తారా?",
    'default': "I'm having trouble with my connection right now. Did you take your medicine this morning?"
}


# --- TTS Function (Must be before LLM function to prevent call errors) ---
def generate_voice_response(client: ElevenLabs, text: str, voice_id: str, lang_code: str) -> bytes:
    if not client:
        return b"MOCK_AUDIO_DATA"
    try:
        audio_stream = client.text_to_speech.convert( text=text, voice_id=voice_id )
    except Exception as e:
        print(f"ElevenLabs API Error (TTS Failed): {e}")
        return b"TTS_API_FAIL"
    audio_bytes = b"".join(audio_stream)
    save(audio_bytes, f"response_{lang_code}.mp3")
    print(f"✅ Generated audio saved: response_{lang_code}.mp3")
    return audio_bytes


# --- CORE DYNAMIC LLM FUNCTION ---
def analyze_and_generate_response(raw_transcript: str, lang_code: str, elder_name: str, history: list) -> dict:
    """
    Primary logic: Tries Gemini, falls back to multilingual hardcoded response on error.
    """
    global gemini_client

    # The system instruction defines the AI's persona
    system_instruction = f"""
    You are a compassionate, empathetic, and responsible family health assistant for {elder_name}, calling on behalf of their granddaughter, Seema. Your persona is warm and brief.

    ***ABSOLUTE HIGHEST PRIORITY RULE***

    1. **CLOSING CHECK:** Analyze the LAST USER MESSAGE. If it contains any closing signal ('अलविदा', 'शुक्रिया', 'bye', 'rest', 'nothing else', 'goodbye', 'आराम करूंगी'), your **ENTIRE RESPONSE MUST BE ONLY A SINGLE, FINAL CLOSING STATEMENT**. 
       Example Closing: 'ठीक है दादी, अपना ख़याल रखिए। मैं फ़ोन रखती हूँ।' (Take care, I am hanging up now.)
       DO NOT ask any further questions, DO NOT summarize, and DO NOT ask about health.

    2. **CONTINUE CONVERSATION:** If no closing signal is detected, generate a compassionate follow-up question related to the elder's health or status.

    Respond gently and naturally ONLY in the language for {lang_code}.
    """

    # 1. Prepare history for Gemini (Converts list of dicts to Gemini API Content objects)
    contents = []
    for message in history:
        # Note: raw_transcript is the last 'user' message, which is already in the history list from db_interface.
        contents.append(types.Content(role=message['role'], parts=[types.Part.from_text(text=message['text'])]))

    # --- ATTEMPT 1: GEMINI (PRIORITY PATH) ---
    if gemini_client:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents, # Sending the full conversation history
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.7)
            )
            return {"response_text": response.text, "intent": "dynamic_gemini_response"}
        except APIError as e:
            # Captures the 429 quota error, forcing hardcoded fallback
            print(f"Gemini API Error (429 Quota): {e}. Falling back to hardcoded response.")

    # --- FALLBACK (If Gemini client is None, or API fails) ---
    # FIX: Uses the globally defined FALLBACK_MAP
    fallback_text = FALLBACK_MAP.get(lang_code, FALLBACK_MAP['default'])
    print(f"WARNING: Using hardcoded fallback: {fallback_text}")

    return {"response_text": fallback_text, "intent": "hardcoded_fallback"}


# ----------------------------------------------------------------------
# --- ONE-OFF GREETING GENERATION (MUST BE AT THE END OF THE FILE) ---
# ----------------------------------------------------------------------

def generate_initial_greeting():
    """Generates initial greeting audio files only if they do not already exist."""

    GREETING_CONFIG = {
        'en-IN': {
            'voice_id': "50YSQEDPA2vlOxhCseP4",
            'text': "Hello Grandma, I'm your health assistant calling on behalf of your granddaughter, Seema. Did you take your blue pill this morning?",
        },
        'hi-IN': {
            'voice_id': "50YSQEDPA2vlOxhCseP4",
            'text': "नमस्ते दादी, मैं आपकी पोती, सीमा की तरफ़ से बोल रही हूँ। क्या आपने आज सुबह 9 बजे वाली दवा ले ली है?",
        },
        'te-IN': {
            'voice_id': "Nda4CxqYPMJ65wadFnhJ",
            'text': "నమస్కారం అమ్మమ్మా, నేను మీ మనవరాలు సీమ తరఫున మాట్లాడుతున్నాను. ఈ రోజు ఉదయం 9 గంటలకు మీ మందు వేశారా?",
        }
    }

    print("\n--- Initial Greeting Audio Check ---")
    for lang_code, config in GREETING_CONFIG.items():
        output_filename = f"response_initial_greeting_{lang_code}.mp3"

        # 1. CHECK: If the file already exists locally, skip generation
        if path.exists(output_filename):
            print(f"✅ SKIPPING: {output_filename} already exists. Credits saved.")
            continue

            # 2. GENERATE: If the file doesn't exist
        if eleven_client:
            print(f"⚠️ Generating new audio for: {lang_code}...")

            try:
                # FIX: Removed model= argument to avoid SDK TypeError. Relies on Voice ID quality.
                audio_stream = eleven_client.text_to_speech.convert(
                    text=config['text'],
                    voice_id=config['voice_id']
                )
                audio_bytes = b"".join(audio_stream)
                save(audio_bytes, output_filename)
                print(f"✅ Generated audio saved: {output_filename}")
            except Exception as e:
                print(f"CRITICAL: ElevenLabs API Error during greeting generation for {lang_code}: {e}")
        else:
            print(f"CRITICAL: Cannot generate {output_filename}. ElevenLabs client is not initialized.")

    print("--- Greeting Audio Check Complete ---")

# Run this function once locally when the script is imported/run
generate_initial_greeting()

print("DEBUG: END OF EMAPTHY ENGINE FILE EXECUTION.")