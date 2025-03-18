"""
API Connection Test Script for Directory Chatbot

This script tests the connection to your Directory Chatbot API and verifies
that the essential endpoints are available and responding correctly.
"""

import requests
import json
import sys
from typing import Dict, Any, Tuple

# Configuration
API_HOST = "http://localhost:8000"  # Your main API server
DEV_API_KEY = "dev-dashboard-key"   # API key for dev dashboard
TEST_PHONE = "+19995551234"         # Test phone number
TEST_MESSAGE = "Hello, this is a test message"

def print_result(test_name: str, success: bool, details: str = ""):
    """Print a formatted test result."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} | {test_name}")
    if details and not success:
        print(f"       {details}")
    print()

def test_api_connection() -> Tuple[bool, str]:
    """Test basic connection to the API server."""
    try:
        response = requests.get(
            f"{API_HOST}/",
            timeout=5
        )
        return response.status_code == 200, f"Status code: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def test_simulate_message() -> Tuple[bool, str]:
    """Test the simulate message endpoint."""
    try:
        response = requests.post(
            f"{API_HOST}/api/simulate/message",
            json={
                "phone_number": TEST_PHONE,
                "message": TEST_MESSAGE
            },
            headers={
                "X-API-Key": DEV_API_KEY,
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        # Check if response is successful
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}, Response: {response.text}"
        
        # Check if response contains expected fields
        data = response.json()
        if "message" not in data:
            return False, f"Response missing 'message' field: {json.dumps(data)}"
        
        return True, ""
    except Exception as e:
        return False, f"Request error: {str(e)}"

def test_get_action_types() -> Tuple[bool, str]:
    """Test the get action types endpoint."""
    try:
        response = requests.get(
            f"{API_HOST}/api/action-types",
            headers={
                "X-API-Key": DEV_API_KEY
            },
            timeout=5
        )
        
        # Check if response is successful
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}, Response: {response.text}"
        
        # Check if response contains action_types
        data = response.json()
        if "action_types" not in data or not isinstance(data["action_types"], list):
            return False, f"Response missing 'action_types' array: {json.dumps(data)}"
        
        return True, ""
    except Exception as e:
        return False, f"Request error: {str(e)}"

def test_get_workflows() -> Tuple[bool, str]:
    """Test the get workflows endpoint."""
    try:
        response = requests.get(
            f"{API_HOST}/api/workflows",
            headers={
                "X-API-Key": DEV_API_KEY
            },
            timeout=5
        )
        
        # Check if response is successful
        if response.status_code != 200:
            return False, f"Status code: {response.status_code}, Response: {response.text}"
        
        # Check if response contains workflows
        data = response.json()
        if "workflows" not in data or not isinstance(data["workflows"], list):
            return False, f"Response missing 'workflows' array: {json.dumps(data)}"
        
        return True, ""
    except Exception as e:
        return False, f"Request error: {str(e)}"

def main():
    """Run all tests and report results."""
    print("\n=== Directory Chatbot API Connection Test ===\n")
    print(f"API Server: {API_HOST}\n")
    
    # Run tests
    tests = [
        ("API Connection", test_api_connection),
        ("Simulate Message", test_simulate_message),
        ("Get Action Types", test_get_action_types),
        ("Get Workflows", test_get_workflows)
    ]
    
    all_passed = True
    for name, test_func in tests:
        success, details = test_func()
        print_result(name, success, details)
        if not success:
            all_passed = False
    
    # Print summary
    print("=" * 40)
    if all_passed:
        print("All tests PASSED! The API is working correctly.")
        print("You can now run the WhatsApp simulator.")
    else:
        print("Some tests FAILED. Please check the error details and fix any issues.")
        print("Make sure your API server is running and configured correctly.")
    
    print("\nNext steps:")
    print("1. Run 'python whatsapp_simulator.py'")
    print("2. Open http://localhost:8080 in your browser")
    print("=" * 40 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())