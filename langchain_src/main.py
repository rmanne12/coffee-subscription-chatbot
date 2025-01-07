import os
from dotenv import load_dotenv
from langchain_src.chains.userflow_chain import create_userflow_chain

def main():
    try:
        load_dotenv()

        chain = create_userflow_chain()
        conversation_history = []
        stage = "initial"
        active_offer = None

        print("\nWelcome to Bean Me Up (Scotty!) Chatbot!")
        print("Type 'quit' or 'exit' to end the conversation.\n")

        print("Bot: Greetings, Earthling! I heard you want to cancel your subscription. What's on your mind?")

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Bot: Understood. Live long and prosper!")
                break

            conversation_history.append(f"User: {user_input}")

            try:
                inputs = {"conversation": "\n".join(conversation_history), "stage": stage}
                if stage == "offer":
                    inputs.update({"reason": reason, "mood": mood, "urgency": urgency})
                elif stage == "confirmation":
                    inputs.update({"offer": active_offer})

                outputs = chain.run(inputs)
                # print(f"DEBUG: Outputs = {outputs}, Type = {type(outputs)}")

                if not isinstance(outputs, dict) or "final_response" not in outputs:
                    raise KeyError(f"Missing expected output keys: {outputs}")

                bot_response = outputs["final_response"]

                if stage == "initial":
                    mood = outputs.get("mood", "neutral")
                    urgency = outputs.get("urgency", "low")
                    reason = outputs.get("reason", "unknown")
                    stage = "offer"
                elif stage == "offer":
                    active_offer = outputs.get("offer")
                    stage = "confirmation"
                elif stage == "confirmation":
                    stage = "complete"  

                print(f"Bot: {bot_response}")
                conversation_history.append(f"Bot: {bot_response}")

            except KeyError as e:
                print(f"Error: {e}")
                print("Bot: Apologies, there was an issue processing your request.")

    except KeyboardInterrupt:
        print("\nBot: Goodbye, Earthling!")
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
