import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
bucket = os.environ.get("SUPABASE_STORAGE_BUCKET", "media")

supabase = create_client(url, key)

file_path = "dummy.txt"
with open(file_path, "w") as f:
    f.write("hello from supabase")

dest_path = "test/dummy.txt"  # inside bucket

with open(file_path, "rb") as f:
    supabase.storage.from_(bucket).upload(
        dest_path,
        f,
        {
            "content-type": "text/plain",
            "x-upsert": "true"     # MUST be a string, not bool
        },
    )

public_url = supabase.storage.from_(bucket).get_public_url(dest_path)
print("Public URL:", public_url)