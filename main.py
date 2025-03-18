"""
Main entry point for the Directory Chatbot application.

This script runs the FastAPI application with appropriate settings
and includes a direct import of the simulator for convenience.
"""

import os
import uvicorn
import argparse
from dotenv import load_dotenv

def run_app(host="127.0.0.1", port=8000, reload=False):
    """Run the main API server."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Import the FastAPI app
    try:
        import app.api.api as api
        print(f"Starting Directory Chatbot API on http://{host}:{port}")
        uvicorn.run("app.api.api:app", host=host, port=port, reload=reload)
    except ImportError:
        print("Error: Could not import the API module.")
        print("Make sure your project structure is correct and all dependencies are installed.")

def run_simulator(host="0.0.0.0", port=8080):
    """Run the WhatsApp simulator."""
    try:
        from whatsapp_simulator import app as simulator_app
        print(f"Starting WhatsApp Simulator on http://{host}:{port}")
        print("Visit this URL in your browser to access the simulator.")
        uvicorn.run(simulator_app, host=host, port=port)
    except ImportError:
        print("Error: Could not import the WhatsApp simulator.")
        print("Make sure whatsapp_simulator.py is in the current directory.")

def main():
    """Parse command line arguments and run the appropriate server."""
    parser = argparse.ArgumentParser(description="Directory Chatbot Server")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run the main API server")
    api_parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    
    # Simulator command
    sim_parser = subparsers.add_parser("simulator", help="Run the WhatsApp simulator")
    sim_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the simulator to")
    sim_parser.add_argument("--port", type=int, default=8080, help="Port to bind the simulator to")
    
    # Both servers command
    both_parser = subparsers.add_parser("both", help="Run both the API server and the simulator")
    both_parser.add_argument("--api-host", default="127.0.0.1", help="Host to bind the API server to")
    both_parser.add_argument("--api-port", type=int, default=8000, help="Port to bind the API server to")
    both_parser.add_argument("--sim-host", default="0.0.0.0", help="Host to bind the simulator to")
    both_parser.add_argument("--sim-port", type=int, default=8080, help="Port to bind the simulator to")
    both_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes (API server only)")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_app(args.host, args.port, args.reload)
    elif args.command == "simulator":
        run_simulator(args.host, args.port)
    elif args.command == "both":
        print("Running both the API server and the simulator...")
        print("Note: This method doesn't support graceful shutdown. Use Ctrl+C to stop both servers.")
        
        # We need to run the API server in a separate process
        import multiprocessing
        api_process = multiprocessing.Process(
            target=run_app, 
            args=(args.api_host, args.api_port, args.reload)
        )
        api_process.start()
        
        # Run the simulator in the main process
        try:
            run_simulator(args.sim_host, args.sim_port)
        except KeyboardInterrupt:
            print("\nStopping servers...")
            api_process.terminate()
            api_process.join()
            print("Servers stopped.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()