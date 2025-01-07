import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def main():
    """
    Entry point for the Bean Me Up chatbot.
    Eventually, this will:
    1. Initialize the supervisor/orchestrator.
    2. Possibly create a loop to accept user input.
    3. Pass user input to the supervisor and get responses.
    """
    print("Hello from Bean Me Up Chatbot. (Main script)")
    # TODO: Implement supervisor logic & user input loop.

if __name__ == "__main__":
    main()
