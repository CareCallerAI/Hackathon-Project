# CareCallerAI Hackathon Project

Welcome to the CareCallerAI Hackathon Project!  
This repository powers the backend for an AI-driven empathy and health companion platform, designed to provide support for elders and their families via intelligent reminders, sentiment analysis, and voice-based interactions.

## Key Features

- **Voice Emotion Analysis:** Analyzes moods and emotions from voice or text input.
- **Reminders & Notifications:** Schedule simple and medication-specific reminders, notifications, and voice calls.
- **Voice Recording/Upload:** Record or upload voice messages to be used for notifications or reminders.
- **Text-to-Speech (TTS):** Generates and delivers empathetic voice responses using TTS technology.
- **Family & Health Worker Integration:** Escalates alerts via automated calls to family members and community health workers if concerning moods or responses are detected.
- **Call Logs & Scheduler:** Maintains logs of calls and debug endpoints to test scheduling.

## Technology Stack

- **Python** (main backend language)
- **Flask** (REST API)
- **FastAPI** (Empathy Core Test API)
- **Supabase** (storage, reminders, logging)
- **Google Cloud** (TTS, Sentiment, Speech-to-Text, Translation)
- **Twilio** (Voice/SMS notifications)
- **Uvicorn** (ASGI server for FastAPI)
- **Dotenv** (environment configuration)

## API Overview

Core REST endpoints include:
- `/process-emotion` — Analyze mood/emotion from voice or text.
- `/reminders`, `/medication-reminders` — Set various types of reminders.
- `/synthesize` — Generate a TTS voice message.
- `/upload-voice` — Upload audio for notifications/reminders.
- `/send-otp`, `/verify-otp` — OTP-based authentication.
- `/family-members` — Manage family contacts.
- `/call-logs` — Retrieve call history.
- `/twilio/med-flow` — Twilio-assisted medication flows.

For detailed usage, see the docstrings and code comments in `backend/app.py` and `backend/main.py`.

## Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/CareCallerAI/Hackathon-Project.git
   cd Hackathon-Project
   ```

2. **Install Python Dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   - Create a `.env` file in `backend/` with the following keys:
     ```
     SUPABASE_URL=your_supabase_url
     SUPABASE_SERVICE_ROLE_KEY=your_service_key
     SUPABASE_STORAGE_BUCKET=media
     OPENAI_API_KEY=your_openai_key
     TWILIO_ACCOUNT_SID=your_twilio_sid
     TWILIO_AUTH_TOKEN=your_twilio_token
     TWILIO_FROM=your_twilio_phone_number
     ```
   - Google Cloud credentials (JSON) must be set up for TTS/Sentiment/Speech services.

4. **Run the Flask Backend:**
   ```bash
   export FLASK_APP=app.py
   flask run --host=0.0.0.0 --port=5000
   ```
   Or run the FastAPI app:
   ```bash
   python main.py
   ```
   (Check code and endpoints for more details.)

## Contribution

PRs are welcome for improvements, bugfixes, documentation, and new features.

## License

Currently, no license is set. Please contact the CareCallerAI team for usage details.
