# backend/firebase_admin_init.py
from dotenv import load_dotenv
load_dotenv()

import os
import json
import firebase_admin
from firebase_admin import credentials, storage, db

# Support either a file path to service account JSON OR the JSON itself in an env var.
service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_PATH")
service_account_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")  # optional: raw JSON

if not service_account_path and not service_account_json:
    raise Exception("Set FIREBASE_SERVICE_ACCOUNT_JSON_PATH or FIREBASE_SERVICE_ACCOUNT_JSON env var")

if service_account_json and not service_account_path:
    # write json to a temp file so firebase_admin can read it
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.write(service_account_json.encode("utf-8"))
    tmp.flush()
    service_account_path = tmp.name

if not os.path.exists(service_account_path):
    raise Exception(f"Service account JSON not found at: {service_account_path}")

cred = credentials.Certificate(service_account_path)

# Avoid initialize_app being called twice (useful during tests / reloader)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "databaseURL": os.getenv("FIREBASE_DATABASE_URL")
    })

bucket = storage.bucket()
rt_db = db.reference()