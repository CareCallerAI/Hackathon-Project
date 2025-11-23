import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase = create_client(url, key)

# simple test: insert into users
from pprint import pprint

user_key = "testuser"
mobile = "+911234567890"

res = supabase.table("users").upsert(
    {"user_key": user_key, "mobile": mobile},
    on_conflict="user_key"
).execute()

print("Upsert result:")
pprint(res.data)

res2 = supabase.table("users").select("*").eq("user_key", user_key).execute()
print("Select result:")
pprint(res2.data)