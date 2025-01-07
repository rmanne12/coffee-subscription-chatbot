import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_src.config.llm_config import get_llm

# Enhanced prompt with urgency definitions and additional clarity
mood_prompt = PromptTemplate(
    input_variables=["conversation"],
    template="""
You are a classification assistant analyzing a conversation to assess the user's current mood and urgency.

### Instructions:
1. Focus on the user's most recent statements in the conversation.
2. Assign one of the following moods based on the user's tone and language:
   - Possible moods: [neutral, frustrated, angry, happy, sad, confused]
3. Determine the urgency level based on the user's statements:
   - Urgency reflects how immediate or critical the user's request or concern is.
   - Possible urgency levels: [low, high]

### Definitions:
1. **High Urgency**:
   - The user requires immediate action or resolution.
   - Often expressed with strong, demanding, or impatient language (e.g., "Cancel my subscription now!", "I need help right away!").
   - Implies that delaying a response could lead to dissatisfaction or escalation.

2. **Low Urgency**:
   - The user is making a general inquiry or expressing a non-critical concern.
   - Language is calm, exploratory, or indicates no immediate need (e.g., "Can you explain how to pause my subscription?", "I might want to cancel in the future.").

### Examples:

1. **User is frustrated and needs immediate resolution**:
   - Conversation: "I’ve been waiting for hours, and no one has responded!"
   - Output:
     {{
       "mood": "frustrated",
       "urgency": "high"
     }}

2. **User is happy and relaxed, expressing gratitude**:
   - Conversation: "Thank you for the great deal! I’m really excited to try it."
   - Output:
     {{
       "mood": "happy",
       "urgency": "low"
     }}

3. **User is confused and seeking clarification**:
   - Conversation: "I’m not sure how this works. Can you explain it to me?"
   - Output:
     {{
       "mood": "confused",
       "urgency": "low"
     }}

4. **User is angry and demanding cancellation immediately**:
   - Conversation: "I’ve had enough! Cancel my subscription right now!"
   - Output:
     {{
       "mood": "angry",
       "urgency": "high"
     }}

5. **User is sad and expressing disappointment, but not demanding action**:
   - Conversation: "I really wish I could afford this, but I can’t right now."
   - Output:
     {{
       "mood": "sad",
       "urgency": "low"
     }}

### Input:
Conversation:
{conversation}

### Output:
Return JSON ONLY in the following format:
{{
  "mood": "<one_of_the_moods>",
  "urgency": "<low_or_high>"
}}
"""
)

def build_mood_chain() -> LLMChain:
    """
    Creates an LLMChain that analyzes mood and urgency based on the conversation.
    """
    llm = get_llm(model_name="gpt-3.5-turbo", temperature=0.0)
    return LLMChain(
        llm=llm,
        prompt=mood_prompt,
        output_key="mood_json"
    )

def parse_mood_output(mood_json: str) -> dict:
    """
    Parses the chain's JSON output to extract mood and urgency details.
    """
    try:
        data = json.loads(mood_json)
        return {
            "mood": data.get("mood", "neutral"),
            "urgency": data.get("urgency", "low")
        }
    except json.JSONDecodeError:
        # Fallback values for parsing errors
        return {"mood": "neutral", "urgency": "low"}
