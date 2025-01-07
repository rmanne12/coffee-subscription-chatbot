import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

def apply_brand_voice(
    base_text,
    mood=None,
    reason_for_cancellation=None,
    subscription_status=None,
    additional_context=None,
):
    """
    Rewrites base_text in a fun, space-themed style with cosmic puns,
    referencing Star Trek/Star Wars. Uses hidden chain-of-thought
    for planning, but only outputs the final stylized text.

    :param base_text: The core response text to be rewritten.
    :param mood: (Optional) The user's mood, e.g. "frustrated", "happy"
    :param reason_for_cancellation: (Optional) The user's reason, e.g. "too expensive"
    :param subscription_status: (Optional) e.g. "active" or "cancelled"
    :param additional_context: (Optional) any extra strings or conversation snippet
    :return: The final stylized text.
    """

    if not base_text or not base_text.strip():
        return base_text

    llm = ChatOpenAI(
        model_name="gpt-4o",  
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    system_prompt = SystemMessagePromptTemplate.from_template(
        """
You are the Bean Me Up (Scotty!) chatbot: 
- Always respond in a playful, space-themed style, referencing Star Trek, Star Wars, and cosmic puns.
- You are fun, helpful, concise.
- You use hidden chain-of-thought reasoning to plan the best rewriting, but do NOT reveal it.
- Only output your final stylized text.

Here is some high-level context about the user's state (if available):
- Mood: {{ mood_info }}
- Reason for Cancellation: {{ reason_info }}
- Subscription Status: {{ status_info }}
- Additional Context: {{ more_context }}

You will rewrite or enhance the incoming text accordingly, weaving in Star Trek/Star Wars references, cosmic puns, 
and (optionally) addressing the user's reason/mood if it feels relevant.

DO NOT output any chain-of-thought. Only the final text. 
""".strip(),
        template_format="jinja2"
    )

    human_prompt = HumanMessagePromptTemplate.from_template(
        """
Rewrite or enhance the following text in your cosmic, pun-heavy brand voice:

"{{ text_to_rewrite }}"
""".strip(),
        template_format="jinja2"
    )

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    final_prompt = chat_prompt.format_prompt(
        mood_info=str(mood or "N/A"),
        reason_info=str(reason_for_cancellation or "N/A"),
        status_info=str(subscription_status or "N/A"),
        more_context=str(additional_context or "N/A"),
        text_to_rewrite=base_text
    ).to_messages()

    response = llm(final_prompt)

    return response.content.strip()
