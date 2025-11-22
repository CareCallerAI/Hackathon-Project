# test_gemini.py
from google import genai
from google.genai import types

API_KEY = "AIzaSyCOn8hrqSgvzn6dk3xHTCKROCObDq2muZ8"

client = genai.Client(api_key=API_KEY)

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Say: 'hello from test script'"],
    config=types.GenerateContentConfig(temperature=0.1),
)

print("RESPONSE:", resp.text)
