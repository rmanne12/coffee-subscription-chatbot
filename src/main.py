import os
from dotenv import load_dotenv
from src.supervisor.supervisor_agent import supervisor_logic

def main():
    load_dotenv()  


    conversation_state = {
        "history": [],
        "reason_for_cancellation": None,
        "offer_made": False,
        "offer_info": {"offers": []},
        "status": None,
        "mood": {}
    }


    print("\nWelcome to Bean Me Up (Scotty!) Chatbot!")
    print("Type 'quit' or 'exit' to end the conversation.\n")

    initial_bot_message = (
        "Greetings, Earthling! I heard you want to cancel your subscription. "
        "What's on your mind?"
    )
    print(f"Bot: {initial_bot_message}\n")
    conversation_state["history"].append({"bot": initial_bot_message})

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["quit", "exit"]:
            print("Bot: Understood. Live long and prosper out there, Captain!")
            break

        bot_response = supervisor_logic(user_input, conversation_state)
        print(f"Bot: {bot_response}\n")

if __name__ == "__main__":
    main()


