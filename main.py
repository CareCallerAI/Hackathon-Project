# main.py (FINAL VERSION FOR PRODUCTION MERGE)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Import all necessary objects, including the clients, the LLM function,
#    AND the final production functions from db_interface.
from empathy_engine import analyze_and_generate_response, generate_voice_response, MOCK_VOICE_ID, eleven_client
from db_interface import read_transcript, save_response_audio # <-- FINAL DB FUNCTIONS
from db_interface import MOCK_DATA_STATE # Used for local testing state only
from config import ENVIRONMENT # Used to confirm environment
# main.py
# ... existing imports ...
# FIX: Import the MOCK functions instead of the PROD functions for now
from db_interface import mock_read_transcript as read_transcript, mock_save_response_audio as save_response_audio
from db_interface import MOCK_DATA_STATE # Need access to mock data state
from db_interface import CONVERSATION_SUBCOLLECTION, FAMILY_COLLECTION # Need access to constants
import config

# --- SETUP ---
app = FastAPI(title="CareCaller Empathy Core Test API")


# Define the data structure for the incoming request (TRIGGER)
# This model now accepts both IDs needed for the production DB path
class CallInput(BaseModel):
    family_id: str # The document ID of the family (e.g., 6GNZ9B3fGM...)
    call_id: str  # The unique ID for the conversation (e.g., LMZa0PHOGp...)


# --- MOCK STATE/DATABASE (Used for reference) ---
# Keeping MOCK_USER_STATE for general reference, but data is managed in db_interface.
MOCK_USER_STATE = {
    "Grandma Sue": {"meds_taken": False, "call_attempts": 1, "last_tone": "neutral"},
}


@app.get("/")
def read_root():
    return {"message": "Empathy Engine Core is Running"}


@app.post("/process_call")
async def process_call(data: CallInput):
    """
    Triggers the AI processing for a specific call ID, reads history, processes, and writes back.
    Uses the production functions that switch between MOCK and PROD based on config.py.
    """

    # 1. READ ELDER RESPONSE & HISTORY (Production Function Call)
    # The read function uses the family_id and call_id to find the data,
    # OR uses the mock data if config.ENVIRONMENT == 'MOCK'.
    call_data = read_transcript(data.family_id, data.call_id)

    if not call_data:
        # NOTE: This error now catches live DB failures AND mock data failures
        return {"error": "Call ID not found in database. Processing aborted."}

    elder_name = call_data.get('elder_name')
    raw_transcript = call_data.get('raw_text')
    lang_code = call_data.get('language_code')
    history = call_data.get('history', [])

    # 2. LLM Analysis and Dynamic Response Generation (AI Brain)
    llm_output = analyze_and_generate_response(
        raw_transcript=raw_transcript,
        lang_code=lang_code,
        elder_name=elder_name,
        history=history
    )
    response_text = llm_output['response_text']

    # 3. Update History (Append AI's response for the next turn)
    history.append({"role": "model", "text": response_text})

    # 4. Generate Voice Response (TTS)
    audio_bytes = generate_voice_response(
        client=eleven_client,
        text=response_text,
        voice_id=MOCK_VOICE_ID,
        lang_code=lang_code
    )

    # 5. SAVE FINAL VOICE/TEXT AND UPDATED HISTORY (Production Function Call)
    save_response_audio(
        family_id=data.family_id,
        call_id=data.call_id,
        audio_bytes=audio_bytes,
        response_text=response_text,
        updated_history=history
    )

    # 6. Check if audio_bytes is valid before calling len()
    if audio_bytes is None or audio_bytes == b"TTS_API_FAIL":
        audio_status = "FAILURE: TTS Generation Failed"
        audio_length = 0
    else:
        audio_status = "SUCCESS"
        audio_length = len(audio_bytes)


    # 7. Return status (Confirmation that processing is complete)
    return {
        "response_text": response_text,
        "detected_source": llm_output.get('intent'),
        "audio_generation_status": f"{audio_status}: {audio_length} bytes generated.",
        "call_id": data.call_id
    }


# --- RUN SERVER BLOCK ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)