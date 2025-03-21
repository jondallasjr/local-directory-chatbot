"""
WhatsApp Simulator Server for Directory Chatbot

This script serves the WhatsApp simulator web interface and proxies API requests
to the main application.
"""

import os
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import httpx

# Create the FastAPI app
app = FastAPI(title="WhatsApp Simulator")

# Add CORS middleware to allow requests from the simulator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the simulator HTML file
SIMULATOR_HTML_PATH = os.path.join(SCRIPT_DIR, "whatsapp_simulator_new.html")

# API server configuration - adjust these to match your setup
API_HOST = "http://localhost:8000"  # Your main API server
DEV_API_KEY = "dev-dashboard-key"   # API key for dev dashboard

@app.get("/", response_class=HTMLResponse)
async def serve_simulator():
    """Serve the WhatsApp simulator HTML."""
    try:
        # Check if the HTML file exists
        if not os.path.exists(SIMULATOR_HTML_PATH):
            # If file doesn't exist, create it with the simulator HTML content
            with open(SIMULATOR_HTML_PATH, "w") as f:
                f.write(SIMULATOR_HTML)
        
        # Read the simulator HTML
        with open(SIMULATOR_HTML_PATH, "r") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Error serving simulator</h1>
                <p>{str(e)}</p>
                <p>Please save the HTML content to a file named "whatsapp_simulator.html" in the same directory as this script.</p>
            </body>
        </html>
        """)

@app.post("/api/simulate/message")
async def simulate_message(request: Request):
    """Proxy endpoint for the simulate message API call."""
    try:
        # Parse the request body
        body = await request.json()
        phone_number = body.get("phone_number")
        message = body.get("message")
        
        if not phone_number or not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Phone number and message are required"}
            )
        
        # Call the actual API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_HOST}/api/simulate/message",
                json={
                    "phone_number": phone_number,
                    "message": message
                },
                headers={
                    "X-API-Key": DEV_API_KEY,
                    "Content-Type": "application/json"
                }
            )
            
            # Return the API response
            return JSONResponse(
                status_code=response.status_code,
                content=response.json() if response.status_code == 200 else {"error": response.text}
            )
    except Exception as e:
        # Handle errors (connection errors, etc.)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing request: {str(e)}"}
        )

# Route API requests to the main server
@app.get("/api/{path:path}")
@app.post("/api/{path:path}")
async def proxy_api(path: str, request: Request):
    """Proxy all other API requests to the main server."""
    try:
        # Get the full URL
        url = f"{API_HOST}/api/{path}"
        
        # Get method, headers, and body from the request
        method = request.method
        headers = dict(request.headers)
        headers["X-API-Key"] = DEV_API_KEY  # Make sure to include API key
        
        # Get query parameters
        params = dict(request.query_params)
        
        # Get request body for POST, PUT, etc.
        body = await request.body() if method in ["POST", "PUT", "PATCH"] else None
        
        # Send request to the main API
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                content=body
            )
            
            # Return the response with the same status code and content
            return JSONResponse(
                status_code=response.status_code,
                content=response.json() if response.status_code == 200 else {"error": response.text}
            )
    except Exception as e:
        # Handle errors (connection errors, etc.)
        return JSONResponse(
            status_code=500,
            content={"error": f"Error proxying request: {str(e)}"}
        )

# HTML content for the simulator
SIMULATOR_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Directory Chatbot Simulator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .chat-header {
            background-color: #075E54;
            color: white;
            padding: 15px;
        }
        .chat-body {
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            background-color: #E5DDD5;
        }
        .chat-footer {
            background-color: #F0F0F0;
            padding: 10px;
        }
        .message {
            border-radius: 10px;
            padding: 8px 15px;
            margin-bottom: 10px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #DCF8C6;
            margin-left: auto;
        }
        .bot-message {
            background-color: white;
        }
        .action-panel {
            border-left: 1px solid #dee2e6;
            padding-left: 15px;
            height: 400px;
            overflow-y: auto;
        }
        .phone-selector {
            margin-bottom: 20px;
        }
        .action-card {
            margin-bottom: 10px;
        }
        .json-display {
            font-family: monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container mt-4 mb-4">
        <h1 class="text-center mb-4">Directory Chatbot WhatsApp Simulator</h1>
        
        <div class="row mb-3">
            <div class="col-md-6">
                <div class="phone-selector">
                    <label for="phoneNumber" class="form-label">Phone Number:</label>
                    <div class="input-group">
                        <input type="text" class="form-control" id="phoneNumber" value="+12345678900">
                        <button class="btn btn-outline-secondary" type="button" id="newPhoneBtn">New User</button>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="d-flex justify-content-end mb-2">
                    <button class="btn btn-primary me-2" id="refreshBtn">Refresh History</button>
                    <button class="btn btn-danger" id="clearBtn">Clear Chat</button>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="chat-container">
                    <div class="chat-header">
                        <h5 class="mb-0">WhatsApp Simulator</h5>
                        <small>Connected as <span id="currentPhone">+12345678900</span></small>
                    </div>
                    <div class="chat-body" id="chatBody">
                        <!-- Messages will be dynamically added here -->
                    </div>
                    <div class="chat-footer">
                        <div class="input-group">
                            <input type="text" class="form-control" id="messageInput" placeholder="Type a message">
                            <button class="btn btn-success" id="sendBtn">Send</button>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <ul class="nav nav-tabs card-header-tabs" id="actionTabs" role="tablist">
                            <li class="nav-item" role="presentation">
                                <button class="nav-link active" id="thinking-tab" data-bs-toggle="tab" data-bs-target="#thinking" type="button" role="tab" aria-controls="thinking" aria-selected="true">System Thinking</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="context-tab" data-bs-toggle="tab" data-bs-target="#context" type="button" role="tab" aria-controls="context" aria-selected="false">Active Context</button>
                            </li>
                            <li class="nav-item" role="presentation">
                                <button class="nav-link" id="actions-tab" data-bs-toggle="tab" data-bs-target="#actions" type="button" role="tab" aria-controls="actions" aria-selected="false">Actions</button>
                            </li>
                        </ul>
                    </div>
                    <div class="card-body">
                        <div class="tab-content" id="actionTabContent">
                            <div class="tab-pane fade show active" id="thinking" role="tabpanel" aria-labelledby="thinking-tab">
                                <div id="thinkingContent" class="action-panel">
                                    <div class="alert alert-info">System thinking will appear here after sending a message.</div>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="context" role="tabpanel" aria-labelledby="context-tab">
                                <div id="contextContent" class="action-panel">
                                    <div class="alert alert-info">Active context will appear here after sending a message.</div>
                                </div>
                            </div>
                            <div class="tab-pane fade" id="actions" role="tabpanel" aria-labelledby="actions-tab">
                                <div id="actionsContent" class="action-panel">
                                    <div class="alert alert-info">Action history will appear here after sending a message.</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const apiUrl = window.location.origin; // Use the current host
            const chatBody = document.getElementById('chatBody');
            const messageInput = document.getElementById('messageInput');
            const sendBtn = document.getElementById('sendBtn');
            const phoneNumber = document.getElementById('phoneNumber');
            const currentPhone = document.getElementById('currentPhone');
            const newPhoneBtn = document.getElementById('newPhoneBtn');
            const clearBtn = document.getElementById('clearBtn');
            const refreshBtn = document.getElementById('refreshBtn');
            const thinkingContent = document.getElementById('thinkingContent');
            const contextContent = document.getElementById('contextContent');
            const actionsContent = document.getElementById('actionsContent');
            
            // Load workflows from the API
            function loadWorkflows() {
                fetch(`${apiUrl}/api/workflows`, {
                    headers: {
                        'X-API-Key': 'dev-dashboard-key'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    console.log("Loaded workflows:", data);
                    // You can do something with the workflows here if needed
                })
                .catch(error => {
                    console.error("Error loading workflows:", error);
                });
            }
            
            // localStorage for message history
            let messageHistory = JSON.parse(localStorage.getItem('chatMessages') || '{}');
            
            // Load existing messages for current phone
            function loadMessages() {
                const phone = phoneNumber.value;
                currentPhone.textContent = phone;
                chatBody.innerHTML = '';
                
                if (messageHistory[phone]) {
                    messageHistory[phone].forEach(msg => {
                        addMessageToChat(msg.content, msg.isUser);
                    });
                    // Scroll to bottom
                    chatBody.scrollTop = chatBody.scrollHeight;
                }
            }
            
            // Add message to chat UI
            function addMessageToChat(message, isUser) {
                const messageDiv = document.createElement('div');
                messageDiv.classList.add('message');
                messageDiv.classList.add(isUser ? 'user-message' : 'bot-message');
                messageDiv.textContent = message;
                chatBody.appendChild(messageDiv);
                
                // Scroll to bottom
                chatBody.scrollTop = chatBody.scrollHeight;
                
                // Save to history
                const phone = phoneNumber.value;
                if (!messageHistory[phone]) {
                    messageHistory[phone] = [];
                }
                messageHistory[phone].push({
                    content: message,
                    isUser: isUser,
                    timestamp: new Date().toISOString()
                });
                localStorage.setItem('chatMessages', JSON.stringify(messageHistory));
            }
            
            // Send message
            async function sendMessage() {
                const message = messageInput.value.trim();
                const phone = phoneNumber.value;
                
                if (!message) return;
                
                // Add user message to chat
                addMessageToChat(message, true);
                
                // Clear input
                messageInput.value = '';
                
                // Update thinking panel
                thinkingContent.innerHTML = '<div class="alert alert-info">Processing message...</div>';
                contextContent.innerHTML = '<div class="alert alert-info">Loading context...</div>';
                actionsContent.innerHTML = '<div class="alert alert-info">Loading actions...</div>';
                
                try {
                    // Send to API
                    const response = await fetch(`${apiUrl}/api/simulate/message`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            phone_number: phone,
                            message: message
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`API error: ${response.status}`);
                    }
                    
                    // Parse the response as JSON
                    const responseText = await response.text();
                    console.log("Response text:", responseText);
                    
                    let responseData;
                    try {
                        responseData = JSON.parse(responseText);
                        console.log("Parsed response data:", responseData);
                    } catch (parseError) {
                        console.error("Error parsing response JSON:", parseError);
                        throw new Error(`Failed to parse response: ${parseError.message}`);
                    }
                    
                    // Add bot response to chat
                    if (responseData && responseData.message) {
                        addMessageToChat(responseData.message, false);
                    } else {
                        console.warn("No message in response data");
                    }
                    
                    // Update thinking panel
                    if (responseData && responseData.thinking) {
                        thinkingContent.innerHTML = `
                            <div class="card action-card">
                                <div class="card-header">System Reasoning</div>
                                <div class="card-body">
                                    <p>${responseData.thinking}</p>
                                </div>
                            </div>
                            <div class="card action-card">
                                <div class="card-header">Current Step</div>
                                <div class="card-body">
                                    <p>${responseData.step || 'Not specified'}</p>
                                </div>
                            </div>
                            <div class="card action-card">
                                <div class="card-header">Selected Action</div>
                                <div class="card-body">
                                    <p>${responseData.action || 'Not specified'}</p>
                                </div>
                            </div>
                        `;
                    } else {
                        thinkingContent.innerHTML = '<div class="alert alert-warning">No thinking data returned</div>';
                    }
                    
                    // Update context panel
                    if (responseData && responseData.context) {
                        contextContent.innerHTML = `
                            <div class="card action-card">
                                <div class="card-header">Active Context</div>
                                <div class="card-body">
                                    <div class="json-display">${JSON.stringify(responseData.context, null, 2)}</div>
                                </div>
                            </div>
                        `;
                    } else {
                        contextContent.innerHTML = '<div class="alert alert-warning">No context data returned</div>';
                    }
                    
                    // For actions, we would ideally fetch the action history from the API
                    // but for simplicity, we'll just show the current action
                    if (responseData && responseData.action) {
                        actionsContent.innerHTML = `
                            <div class="card action-card">
                                <div class="card-header">Latest Action</div>
                                <div class="card-body">
                                    <p><strong>Type:</strong> ${responseData.action}</p>
                                    <p><strong>Description:</strong> ${responseData.step || 'No description'}</p>
                                </div>
                            </div>
                        `;
                    } else {
                        actionsContent.innerHTML = '<div class="alert alert-warning">No action data returned</div>';
                    }
                    
                } catch (error) {
                    console.error('Error sending message:', error);
                    thinkingContent.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                    addMessageToChat(`Error: Unable to get response from server. ${error.message}`, false);
                }
            }
            
            // Event listeners
            sendBtn.addEventListener('click', sendMessage);
            
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            phoneNumber.addEventListener('change', function() {
                loadMessages();
            });
            
            newPhoneBtn.addEventListener('click', function() {
                // Generate random phone number
                const randomPhone = '+1' + Math.floor(Math.random() * 10000000000).toString().padStart(10, '0');
                phoneNumber.value = randomPhone;
                loadMessages();
            });
            
            clearBtn.addEventListener('click', function() {
                const phone = phoneNumber.value;
                if (messageHistory[phone]) {
                    delete messageHistory[phone];
                    localStorage.setItem('chatMessages', JSON.stringify(messageHistory));
                    chatBody.innerHTML = '';
                }
            });
            
            refreshBtn.addEventListener('click', function() {
                loadMessages();
            });
            
            // Initial load
            loadMessages();
        });
    </script>
</body>
</html>"""

if __name__ == "__main__":
    # Run the server
    print("Starting WhatsApp Simulator Server...")
    print("API Server URL:", API_HOST)
    print("Visit http://localhost:8080 to access the simulator")
    uvicorn.run(app, host="0.0.0.0", port=8080)