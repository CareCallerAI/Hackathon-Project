# config.py

# --- ENVIRONMENT SWITCH ---
# Set this variable to 'MOCK' for local development/simulation,
# or to 'PROD' for live database connection.

ENVIRONMENT = 'MOCK' # <--- THIS IS YOUR SIMPLE TOGGLE

# --- PROD FIREBASE/FIRESTORE CONSTANTS ---
# These are the actual live IDs your friend will provide.
# We define them here so the logic files (db_interface) can access them.

# Replace these placeholders with your friend's confirmed live data:
LIVE_FAMILY_ID = "[YOUR_LIVE_FAMILY_ID]"
ELDER_NAME_FIELD = "[ELDER_NAME_FIELD]"
RAW_TEXT_FIELD = "[RAW_TEXT_FIELD]"
HISTORY_ARRAY_FIELD = "[HISTORY_ARRAY_FIELD]"

# NOTE: The actual collection paths are defined in db_interface.py