import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_src.config.llm_config import get_llm

VALID_REASONS = [
    "too expensive",
    "stockpiled",
    "too jittery",
    "not jittery enough",
    "taste",
    "moving"
]

# Enhanced reason prompt with detailed guidance
reason_prompt = PromptTemplate(
    input_variables=["conversation", "valid_reasons"],
    template="""
You are an assistant that identifies the user's primary reason for canceling their subscription.

### Valid reasons for cancellation:
{valid_reasons}

### Instructions:
1. Carefully read the conversation to understand the user's concerns and the context of their request.
2. Focus on identifying the specific reason for cancellation. Choose the closest match from the valid reasons provided.
3. If the user's concern doesn't explicitly match any reason, infer the closest valid reason based on their language and tone.

### Examples:

#### Example 1: Too expensive
**Conversation**: 
"I can't afford rent right now. The subscription is too costly for me."

**Output**:
{{
  "reason_for_cancellation": "too expensive"
}}

#### Example 2: Stockpiled
**Conversation**: 
"I already have way too much coffee stockpiled at home. I don’t need any more for now."

**Output**:
{{
  "reason_for_cancellation": "stockpiled"
}}

#### Example 3: Too jittery
**Conversation**: 
"The coffee makes me feel jittery all day. It’s too much for me."

**Output**:
{{
  "reason_for_cancellation": "too jittery"
}}

#### Example 4: Moving
**Conversation**: 
"I’m moving to another city and don’t want the subscription right now."

**Output**:
{{
  "reason_for_cancellation": "moving"
}}

### Input:
Conversation:
{conversation}

### Output:
Return JSON ONLY in this format:
{{
  "reason_for_cancellation": "<one_of_the_valid_reasons>"
}}
"""
)

def build_reason_chain() -> LLMChain:
    """
    Creates an LLMChain for identifying the user's reason for cancellation.
    """
    llm = get_llm(model_name="gpt-3.5-turbo", temperature=0.0)
    return LLMChain(
        llm=llm,
        prompt=reason_prompt,
        output_key="reason_json"
    )

def parse_reason_output(reason_json: str) -> str:
    """
    Parses the chain's JSON output, extracting the reason for cancellation.
    """
    try:
        data = json.loads(reason_json)
        reason = data.get("reason_for_cancellation")
        if reason not in VALID_REASONS:
            return None
        return reason
    except json.JSONDecodeError:
        # Return None if the parsing fails
        return None
