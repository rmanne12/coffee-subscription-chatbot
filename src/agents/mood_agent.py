import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

def determine_mood_and_urgency(conversation_history):
    """
    Analyze the conversation history to determine the user's mood and urgency.
    Returns:
      {
        "mood": "neutral"|"frustrated"|"angry"|"happy"|...,
        "urgency": "low"|"high"
      }
    """

    user_text_combined = "\n".join(
        msg["user"] for msg in conversation_history if "user" in msg
    ).strip()

    bot_text_combined = "\n".join(
        msg["bot"] for msg in conversation_history if "bot" in msg
    ).strip()

    if not user_text_combined:
        return {"mood": "neutral", "urgency": "low"}

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        """
You are a classification assistant. You use hidden reasoning to decide the user's mood and urgency 
based on their responses in the conversation history and their interaction with the bot. 
However, you only output JSON with fields "mood" and "urgency." 
Never reveal your reasoning.

Possible moods: "neutral", "frustrated", "angry", "happy", "sad", "confused", etc.
Possible urgency: "low" or "high".

Consider the most recent user message for context. If the user has accepted an offer or is positive,
the mood is likely "happy" or "neutral." If the user continues to express dissatisfaction, 
consider "frustrated," "angry," or other appropriate moods.

Return valid JSON only, in this format:
{
  "mood": "<one_word_mood>",
  "urgency": "<low_or_high>"
}
""",
        template_format="jinja2"
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """
Conversation so far:
{{ conversation }}

Classify the user's current mood and urgency based on the entire conversation. 
Focus on the most recent interaction for context. 
Use hidden reasoning but return only JSON.
""",
        template_format="jinja2"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    final_prompt = chat_prompt.format_prompt(
        conversation=f"User: {user_text_combined}\nBot: {bot_text_combined}"
    ).to_messages()

    response = llm(final_prompt)
    try:
        data = json.loads(response.content)
        mood = data.get("mood", "neutral")
        urgency = data.get("urgency", "low")
        return {"mood": mood, "urgency": urgency}
    except Exception as e:
        print(f"Error parsing mood and urgency: {e}")
        return {"mood": "neutral", "urgency": "low"}
