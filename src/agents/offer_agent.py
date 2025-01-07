import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

VALID_OFFERS = [
    "50% discount",
    "pause up to 6 months",
    "skip next order"
]

def propose_offer(reason_for_cancellation, mood_info):
    """
    Use hidden chain-of-thought to decide the best offer from:
      - 50% discount
      - pause up to 6 months
      - skip next order
    Return a JSON with:
    {
      "offer_type": "<one_of_these_offers>",
      "offer_text": "<explanation>",
      "offer_accepted": null
    }
    """

    reason_str = reason_for_cancellation or "unknown"
    mood_str = mood_info.get("mood", "neutral")
    urgency_str = mood_info.get("urgency", "low")

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        """
You must choose exactly ONE of the following offers to retain a coffee subscription:
{{ valid_offers }}

Use hidden chain-of-thought reasoning to figure out the best match, 
based on user's reason and mood, but only output valid JSON with these fields:
{
  "offer_type": "<one_of_these>",
  "offer_text": "<concise_explanation>",
  "offer_accepted": null
}
Never reveal your chain-of-thought. Output only the final JSON.
""",
        template_format="jinja2"
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """
User's reason for cancellation: {{ reason }}
User's mood: {{ mood }}
User's urgency: {{ urgency }}

Which single offer is best, and how to present it in a short user-facing message?
Only return JSON.
""",
        template_format="jinja2"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    final_prompt = chat_prompt.format_prompt(
        valid_offers=VALID_OFFERS,
        reason=reason_str,
        mood=mood_str,
        urgency=urgency_str
    ).to_messages()

    response = llm(final_prompt)
    try:
        parsed = json.loads(response.content)
        if parsed.get("offer_type") not in VALID_OFFERS:
            parsed["offer_type"] = "50% discount"
            parsed["offer_text"] = "We can offer 50% off your next order to help lower costs."
        return {
            "offer_type": parsed["offer_type"],
            "offer_text": parsed["offer_text"],
            "offer_accepted": None
        }
    except:
        return {
            "offer_type": "50% discount",
            "offer_text": "We can offer 50% off your next order to help lower costs.",
            "offer_accepted": None
        }
