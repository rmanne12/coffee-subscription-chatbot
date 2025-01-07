import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

def finalize_cancellation_or_accept_offer(conversation_history, offer_info):
    """
    Reads the latest user message to see if the user accepted the offer.
    If accepted => subscription remains active, offer_accepted = True.
    Otherwise => canceled, offer_accepted = False.

    Return:
    {
      "status": "active" | "cancelled",
      "offer_accepted": true | false,
      "confirmation_text": "..."
    }
    """

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    user_text = ""
    for msg in reversed(conversation_history):
        if "user" in msg:
            user_text = msg["user"]
            break

    system_prompt = SystemMessagePromptTemplate.from_template(
        """
You are an assistant deciding if the user accepted or rejected the offer.
Use hidden reasoning, but output only JSON:

{
  "accepted_offer": true or false
}

If user clearly says yes or is positive, accepted_offer = true.
Otherwise, accepted_offer = false.
""",
        template_format="jinja2"
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """
Offer info: {{ offer_info }}
Latest user message: "{{ user_text }}"

Has the user accepted the offer?
Only output JSON with 'accepted_offer': true or false.
""",
        template_format="jinja2"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    final_prompt = chat_prompt.format_prompt(
        offer_info=offer_info,
        user_text=user_text
    ).to_messages()

    response = llm(final_prompt)
    try:
        parsed = json.loads(response.content)
        accepted = parsed.get("accepted_offer", False)
    except:
        accepted = False

    if accepted:
        return {
            "status": "active",
            "offer_accepted": True,
            "confirmation_text": (
                "Mission accomplished! Your subscription remains active under the new plan. "
                "Live long and caffeinate!"
            )
        }
    else:
        return {
            "status": "cancelled",
            "offer_accepted": False,
            "confirmation_text": (
                "Understood. Your subscription is now cancelled. "
                "We’ll be here if you decide to rejoin the cosmic coffee journey!"
            )
        }