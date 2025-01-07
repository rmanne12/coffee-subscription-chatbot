import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

VALID_REASONS = [
    "too expensive",
    "stockpiled",
    "too jittery",
    "not jittery enough",
    "taste",
    "moving"
]

def find_cancellation_reason(conversation_history):
    """
    Identify the user's reason for cancellation from:
      - "too expensive"
      - "stockpiled"
      - "too jittery"
      - "not jittery enough"
      - "taste"
      - "moving"
    
    Returns: { "reason_for_cancellation": <reason> }
    """

    user_text_combined = "\n".join(
        msg["user"] for msg in conversation_history if "user" in msg
    ).strip()
    if not user_text_combined:
        return {"reason_for_cancellation": None}

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        """
You are an assistant that determines which of these 6 reasons applies to a user's cancellation request:
{{ valid_reasons }}

Use hidden reasoning to figure out the best match, but only output JSON in this format:
{
  "reason_for_cancellation": "<one_of_these>"
}

If uncertain, pick the closest match from the valid reasons. 
Do NOT reveal chain-of-thought to the user.
""",
        template_format="jinja2"
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """
The user's conversation so far is:
{{ conversation }}

Identify which reason best matches their intent to cancel. Output only JSON.
""",
        template_format="jinja2"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    final_prompt = chat_prompt.format_prompt(
        valid_reasons=VALID_REASONS,
        conversation=user_text_combined
    ).to_messages()

    response = llm(final_prompt)

    try:
        parsed = json.loads(response.content)
        reason = parsed.get("reason_for_cancellation")
        if reason not in VALID_REASONS:
            reason = None
        return {"reason_for_cancellation": reason}
    except:
        return {"reason_for_cancellation": None}
