# WhatsApp Simulator for Directory Chatbot

This is a development tool for testing and debugging the Directory Chatbot without requiring a real WhatsApp integration. It provides a simulated WhatsApp interface along with developer panels to inspect system reasoning, active context, and actions.

## Features

- Simulated WhatsApp chat interface
- Support for multiple simulated phone numbers/users
- Local chat history persistence
- Developer panels showing:
  - System reasoning and thinking
  - Current workflow step and action
  - Active context data
  - Action history
- Graph knowledge base for storing and retrieving information about entities and their relationships

## Setup

### Basic Setup

1. Make sure your Directory Chatbot API is running (typically on port 8000)
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the simulator:

```bash
python whatsapp_simulator.py
```

4. Open your browser and navigate to: http://localhost:8080

### Database Setup

The application uses Supabase with PostgreSQL for data storage, including a graph knowledge base. Follow these steps to set up the database:

1. Make sure you have a Supabase account and project set up
2. Update the `.env` file with your Supabase credentials:

```
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-key
```

3. Set up the database schema by running the SQL migrations:
   - Log in to your Supabase dashboard
   - Navigate to the SQL Editor
   - Copy and paste the contents of `app/database/migrations/001_setup_graph_kb.sql`
   - Execute the script

4. Verify the database setup by running:

```bash
python setup_graph_kb_direct.py --check-connection
```

5. (Optional) Add sample data to the knowledge base:

```bash
python setup_graph_kb_direct.py --add-sample-data
```

## Knowledge Base Structure

The graph knowledge base has the following components:

- **Entities**: Represent real-world objects (providers, services, events, etc.)
- **Relationships**: Connect entities with typed relationships (offers, provides, organizes, etc.)
- **Vector Embeddings**: Enable semantic search capabilities
- **Graph Traversal**: Allow exploring connected information

### Entity Types

- `provider`: Organizations or individuals providing services
- `service`: Services offered to users
- `product`: Physical items available to users
- `event`: Time-bound activities or gatherings
- `location`: Physical places with geographical coordinates

## Configuration

By default, the simulator is configured to proxy requests to your main API server at `http://localhost:8000`. If your API is running on a different port or hostname, edit the `API_HOST` variable in `whatsapp_simulator.py`:

```python
# API server configuration - adjust these to match your setup
API_HOST = "http://localhost:8000"  # Your main API server
DEV_API_KEY = "dev-dashboard-key"   # API key for dev dashboard
```

## Usage

### Testing Different Users

1. Enter a phone number in the "Phone Number" field to simulate a specific user
2. Click "New User" to generate a random phone number for testing new user experiences

### Managing Conversations

- Type a message in the input field and press "Send" or hit Enter
- Use "Clear Chat" to remove the conversation history for the current phone number
- "Refresh History" reloads the chat history from browser storage

### Developer Panels

The right side of the interface contains developer panels with three tabs:

1. **System Thinking**: Shows the reasoning process, current workflow step, and selected action
2. **Active Context**: Displays the active conversation context data
3. **Actions**: Shows information about the latest executed action

## How It Works

The simulator consists of two main components:

1. **Frontend Interface**: A WhatsApp-style chat UI with developer panels
2. **Proxy Server**: Routes API requests from the frontend to your main API server

When you send a message, the simulator:

1. Stores the message in browser localStorage
2. Sends the message to the `/api/simulate/message` endpoint on your main API
3. Displays the response and any debugging information returned

## Integration with Your API

For best results, your API's `/api/simulate/message` endpoint should return not just the message response, but also debugging information such as:

```json
{
  "message": "Hello, how can I help you?",
  "thinking": "Determining appropriate greeting based on user message",
  "step": "Analyze user message to understand intent",
  "action": "Send User Message",
  "context": {
    "user_intent": "greeting",
    "detected_entities": []
  }
}
```

## Troubleshooting

### API Connection Issues

If you see connection errors in the simulator:

1. Check that your main API server is running
2. Verify the `API_HOST` in `whatsapp_simulator.py` matches your API server address
3. Ensure your API server accepts requests from localhost:8080 (CORS settings)

### Empty or Incomplete Responses

If responses are missing debugging information:

1. Ensure your `/api/simulate/message` endpoint returns the expected JSON format
2. Check browser console for any JavaScript errors
3. Use the browser's network inspector to examine the raw API responses

### Database Connection Issues

If you encounter database connection problems:

1. Verify your Supabase URL and API key in the `.env` file
2. Check that your IP is allowed in the Supabase dashboard
3. Make sure the database schema has been properly set up using the migration files
4. Run `python setup_graph_kb_direct.py --check-connection` to diagnose issues

## Extending the Simulator

The simulator HTML and JavaScript can be modified to add additional features:

1. Edit the `SIMULATOR_HTML` variable in `whatsapp_simulator.py`
2. Add additional tabs, panels, or functionality as needed
3. Restart the simulator server to see your changes

## Development Notes

- Chat history is stored in browser localStorage and persists between sessions
- The simulator runs on a separate port (8080) from your main API server
- API requests are proxied through the simulator to handle CORS issues
- The graph knowledge base uses PostgreSQL with pgvector for semantic search capabilities