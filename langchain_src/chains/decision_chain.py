import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_src.config.llm_config import get_llm

decision_prompt = PromptTemplate(
    input_variables=["conversation", "offers_so_far"],
    template="""
You are a retention specialist tasked with finalizing the user's subscription decision based on their conversation and the offers provided.

### Context:
1. The conversation history with the user is provided below.
2. The list of offers made so far is included.

### Instructions:
1. Analyze the conversation to determine:
   - If the user accepted or declined the offers.
   - Whether the subscription should remain "active" or be "cancelled".
2. Match the following:
   - If the user explicitly accepts an offer, set "offer_accepted" to true and "status" to "active".
   - If the user insists on cancellation, set "offer_accepted" to false and "status" to "cancelled".
   - If undecided, set "offer_accepted" to null and "status" to "active".
3. Provide a concise, user-friendly summary in "confirmation_text".

### Input:
- Conversation: {conversation}
- Offers so far: {offers_so_far}

### Output:
Return JSON ONLY:
{{
  "status": "<active|cancelled>",
  "offer_accepted": <true|false|null>,
  "confirmation_text": "<short user-facing text>"
}}
"""
)

def build_decision_chain() -> LLMChain:
    llm = get_llm("gpt-3.5-turbo", temperature=0.0)
    return LLMChain(
        llm=llm,
        prompt=decision_prompt,
        output_key="decision_json"
    )

def parse_decision_output(decision_json: str) -> dict:
    try:
        data = json.loads(decision_json)
        return {
            "status": data.get("status", "cancelled"),
            "offer_accepted": data.get("offer_accepted", False),
            "confirmation_text": data.get(
                "confirmation_text", "Your subscription has been cancelled."
            )
        }
    except json.JSONDecodeError:
        return {
            "status": "cancelled",
            "offer_accepted": False,
            "confirmation_text": "Your subscription has been cancelled."
        }
