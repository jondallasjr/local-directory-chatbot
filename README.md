# Community Directory Chatbot (MVP 0.01)

A WhatsApp-based community directory service built using LangChain, Groq, Supabase, and Twilio.

## Project Overview

This MVP implements a development-focused control dashboard for a community directory service accessible via WhatsApp. The system enables users to find, add, and edit information about local providers, products, services, events, and community notes through natural language interactions.

## Core Features (MVP 0.01)

- Basic conversation framework with LLM integration
- Simple entity creation and retrieval
- WhatsApp integration via Twilio
- Developer dashboard for monitoring conversations
- Workflow engine with basic action types
- Logging and error handling

## Prerequisites

- Python 3.8+
- PostgreSQL database (via Supabase)
- Twilio account with WhatsApp sandbox
- Groq API key for LLM access

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd local-directory-chatbot

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root with the following variables:

```
# LLM
GROQ_API_KEY=your_groq_api_key

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Messaging
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=your_twilio_whatsapp_number

# Application
DEBUG=True
PORT=8000
```

### 3. Database Setup

1. Create a new project in Supabase
2. Go to the SQL Editor in Supabase dashboard
3. Run the database setup script from `database-setup.sql`
4. Make sure to enable the pgvector extension

### 4. Running the Application

```bash
# Start the application
python main.py
```

The API will be available at `http://localhost:8000` and Swagger documentation at `http://localhost:8000/docs`.

### 5. Twilio WhatsApp Setup

1. Set up a Twilio WhatsApp sandbox
2. Configure the webhook URL to point to your API endpoint:
   - If running locally, use ngrok to expose your local server
   - Set the webhook URL to `https://your-ngrok-url/api/webhook/twilio`
   - Method: POST

## Developer Dashboard

For MVP 0.01, the developer dashboard is accessed via API endpoints:

- `/api/users` - List all users
- `/api/messages/{user_id}` - Get messages for a user
- `/api/workflows/{user_id}` - Get workflows for a user
- `/api/actions/{workflow_id}` - Get actions for a workflow
- `/api/logs` - Get system logs
- `/api/entities` - Search entities
- `/api/simulate/message` - Simulate a message for testing

API requests require an `x-api-key` header with value `dev-dashboard-key` (for MVP 0.01).

## Project Structure

```
app/
├── api/               # API endpoints and Twilio integration
├── config/            # Configuration settings
├── database/          # Database client and operations
├── models/            # LLM integration
├── utils/             # Utility functions
└── workflows/         # Workflow definitions and action handling
```

## Testing the Application

```bash
# Simulate a message
curl -X POST http://localhost:8000/api/simulate/message \
  -H "Content-Type: application/json" \
  -H "x-api-key: dev-dashboard-key" \
  -d '{"phone_number": "+1234567890", "message": "Hello, I want to add a new hairdresser"}'
```

## Development Roadmap

This is MVP 0.01 - Core Conversational Framework. Future development stages:

- MVP 0.1: Basic Entity Management
- MVP 0.25: Primary Business Workflows
- MVP 0.5: Enhanced Search and Business Logic
- MVP 0.75: Pre-Release Enhancement
- MVP 1.0: Production Ready

## License

[Your license information]