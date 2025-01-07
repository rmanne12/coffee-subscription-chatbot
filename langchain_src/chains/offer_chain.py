import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_src.config.llm_config import get_llm

VALID_OFFERS = [
    "50% off next order",
    "pause up to 6 months",
    "skip next order",
    "combo"
]

offer_prompt = PromptTemplate(
    input_variables=["reason", "mood", "urgency", "existing_offers"],
    template="""
You are a retention specialist tasked with analyzing user concerns and providing tailored offers to retain their subscription.

### User Context:
- Reason for cancellation: "{reason}" (e.g., "too expensive", "stockpiled", "taste", etc.)
- Mood: "{mood}" (e.g., "neutral", "angry", "sad", etc.)
- Urgency: "{urgency}" (High: immediate action needed; Low: user is exploring options)
- Existing offers: {existing_offers}

### Offer Categories:
1. **50% off next order** - Suitable for users concerned about cost or willing to try alternative products.
2. **Pause up to 6 months** - Best for users who have stockpiled or face temporary financial constraints.
3. **Skip next order** - Ideal for users needing a short break (e.g., due to moving, traveling).
4. **Combo** - A combination of offers tailored to the user's unique needs.

### Instructions:
1. Match the user's reason, mood, and urgency to the most relevant offer(s).
2. For high urgency or angry/frustrated moods, prioritize quick and direct solutions.
3. Personalize the offer text to resonate with the user and acknowledge their situation.
4. If no specific offer fits, default to "50% off next order" with an empathetic tone.

### Examples:
#### Example 1: Too Expensive
- Mood: Neutral | Urgency: Low
- Offer: "50% off next order"
- Text: "We understand cost concerns. Let’s make your next order 50% off to help you enjoy your coffee at a reduced price!"

#### Example 2: Stockpiled
- Mood: Happy | Urgency: Low
- Offer: "Pause up to 6 months"
- Text: "It seems you’re well-stocked! We’ve paused your subscription for 6 months, so you’ll have plenty of time to enjoy your current stash."

#### Example 3: Moving
- Mood: Frustrated | Urgency: High
- Offer: "Skip next order"
- Text: "We understand moving can be stressful. We’ve skipped your next order to make things easier for you during this transition."

### Input:
- User Context: Reason = "{reason}", Mood = "{mood}", Urgency = "{urgency}", Existing Offers = {existing_offers}

### Output:
Return JSON ONLY in this format:
{{
  "offer_type": "<50% off next order | pause up to 6 months | skip next order | combo>",
  "offer_text": "<short user-facing explanation>",
  "offer_accepted": null
}}
"""
)

def build_offer_chain() -> LLMChain:
    llm = get_llm("gpt-3.5-turbo", temperature=0.2)
    return LLMChain(
        llm=llm,
        prompt=offer_prompt,
        output_key="offer_json"
    )

def parse_offer_output(offer_json: str) -> dict:
    try:
        data = json.loads(offer_json)
        if data.get("offer_type") not in VALID_OFFERS:
            data["offer_type"] = "50% off next order"
            data["offer_text"] = "We understand your concerns. Let’s make your next order 50% off to help out!"
        data["offer_accepted"] = None
        return data
    except json.JSONDecodeError:
        return {
            "offer_type": "50% off next order",
            "offer_text": "We understand your concerns. Let’s make your next order 50% off to help out!",
            "offer_accepted": None
        }
