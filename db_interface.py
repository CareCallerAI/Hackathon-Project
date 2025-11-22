# db_interface.py (FINAL SWITCHABLE VERSION)

from google.cloud import firestore
import os
from google.cloud.firestore_v1.base_client import BaseClient
from os import path
import config
# --- CONSTANTS FROM CONFIG ---
FAMILY_COLLECTION = "family"
CONVERSATION_SUBCOLLECTION = "conversations" # Confirmed path
# Note: Other field names (ELDER_NAME_FIELD, etc.) are imported via config.

# --- FIRESTORE CLIENT INITIALIZATION (Runs only once) ---
try:
    # Use the authenticated Service Account Key (google-service-key.json)
    firestore_db: BaseClient = firestore.Client()
    print("DEBUG: Firestore client initialized.")
except Exception as e:
    print(f"CRITICAL: Firestore client failed to initialize: {e}")
    firestore_db = None


# --- MOCK DATA (In-Memory State for 'MOCK' Environment) ---

# This simulates the data your friend would save/read from the live DB
MOCK_DATA_STATE = {
    # This ID is the standard trigger for the demo
    "DEMO_CALL_ID": {
        "language_code": "hi-IN",
        "elder_name": "Grandma Seema (MOCK)",
        # db_interface.py (Inside MOCK_DATA_STATE["DEMO_CALL_ID"])

        "history": [
            {"role": "model", "text": "नमस्ते दादी, मैं आपकी पोती, सीमा की तरफ़ से बोल रही हूँ। क्या आपने आज सुबह 9 बजे वाली दवा ले ली है?"},
            {"role": "user", "text": "हाँ, मैंने आज सुबह दवाई ले ली है। आप कैसी हैं?"},
            {"role": "model", "text": "मैं तो ठीक हूँ, धन्यवाद पूछने के लिए। लेकिन आज आपको कैसा महसूस हो रहा है? क्या कोई तकलीफ़ है?"},
            {"role": "user", "text": "आज मुझे थोड़ी थकान महसूस हो रही है। इसके अलावा कोई खास बात नहीं है।"},
            {"role": "model", "text": "ओह, आपको थकान क्यों महसूस हो रही है? कृपया आराम करें। अगर आपको कुछ और चाहिए, तो मुझे बताएं।"} # <--- THE AI'S FINAL CONCERN
        ],
        # The final raw_text is the user's goodbye
        "raw_text": "नहीं, मुझे कुछ नहीं चाहिए। मैं अब थोड़ा आराम करूँगी। पूछने के लिए शुक्रिया। अलविदा बेटी। Bye Bye",
    }
}

# --- PRODUCTION / MOCK ROUTER FUNCTIONS ---

def read_transcript(family_id: str, call_id: str) -> dict:
    """Reads the current turn's data, either from Firestore (PROD) or local memory (MOCK)."""

    if config.ENVIRONMENT == 'MOCK':
        # --- MOCK READ PATH ---
        print(f"DEBUG MOCK: Reading data for {call_id} from local memory.")
        # We always read the static DEMO_CALL_ID for local testing continuity
        return MOCK_DATA_STATE.get("DEMO_CALL_ID")

    else:
        # --- PROD FIREBASE READ PATH ---
        if not firestore_db:
            print("CRITICAL PROD ERROR: Firestore DB not available.")
            return None

        doc_ref = (firestore_db.collection(FAMILY_COLLECTION).document(family_id)
                   .collection(CONVERSATION_SUBCOLLECTION).document(call_id))

        doc = doc_ref.get()

        if doc.exists:
            data = doc.to_dict()
            # We use the field names defined in config.py
            return {
                "raw_text": data.get(config.RAW_TEXT_FIELD, ""),
                "language_code": data.get("language_code", "hi-IN"),
                "elder_name": data.get(config.ELDER_NAME_FIELD, "Grandma"),
                "history": data.get(config.HISTORY_ARRAY_FIELD, [])
            }
        return None


def save_response_audio(family_id: str, call_id: str, response_text: str, updated_history: list):
    """Saves the final AI response and updated history, either to Firestore or local memory."""

    # --- MOCK SAVE PATH ---
    if config.ENVIRONMENT == 'MOCK':
        global MOCK_DATA_STATE
        MOCK_DATA_STATE[call_id] = {
            "raw_text": response_text, # Simulate saving AI text for the next read
            "language_code": "hi-IN", # Assumed language for the mock
            "elder_name": MOCK_DATA_STATE.get(call_id, {}).get('elder_name', 'Grandma Seema'),
            "history": updated_history
        }
        print(f"DEBUG MOCK: Saved {len(updated_history)} history items to local memory.")
        return True

    # --- PROD FIREBASE SAVE PATH ---
    else:
        if not firestore_db:
            print("CRITICAL PROD ERROR: Firestore DB not available for saving.")
            return False

        doc_ref = (firestore_db.collection(FAMILY_COLLECTION).document(family_id)
                   .collection(CONVERSATION_SUBCOLLECTION).document(call_id))

        doc_ref.set({
            "status": "processing_complete",
            "ai_response_text": response_text,
            config.HISTORY_ARRAY_FIELD: updated_history, # Use field name from config.py
            "timestamp_completed": firestore.SERVER_TIMESTAMP,
        }, merge=True)

        print(f"DEBUG PROD: Saved final response log to Firestore for {call_id}.")
        return True

    # --- MOCK WRAPPER FUNCTIONS (Used by main.py until fully merged) ---

def mock_read_transcript(family_id: str, call_id: str) -> dict: # <--- ADD family_id HERE
    """TEMP: Simulates reading the elder's data from the mock structure."""
    print(f"DEBUG: Reading data for {call_id} from mock DB...")
    # The body remains the same, as we only need the call_id for the lookup
    return MOCK_DATA_STATE.get("DEMO_CALL_ID")

# db_interface.py (FIXED FINAL mock_save_response_audio signature)

def mock_save_response_audio(family_id: str, call_id: str, audio_bytes: bytes, response_text: str, updated_history: list):
    """TEMP: Simulates saving the final response and history to the mock structure."""
    global MOCK_DATA_STATE

    # The body remains the same, but now it accepts the audio_bytes argument without crashing

    # FIX: Update the in-memory state for the next turn
    if "DEMO_CALL_ID" in MOCK_DATA_STATE:
        MOCK_DATA_STATE["DEMO_CALL_ID"]['history'] = updated_history
        MOCK_DATA_STATE["DEMO_CALL_ID"]['raw_text'] = updated_history[-1]['text']

    print(f"DEBUG: Successfully handled {len(audio_bytes)} bytes of voice message.")
    print(f"DEBUG: Saved final response and {len(updated_history)} history items to mock DB.")
    return True