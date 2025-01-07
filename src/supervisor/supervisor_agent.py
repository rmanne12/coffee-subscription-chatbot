# src/supervisor/supervisor_agent.py

from src.agents.mood_agent import determine_mood_and_urgency
from src.agents.reason_agent import find_cancellation_reason
from src.agents.offer_agent import propose_offer
from src.agents.decision_agent import finalize_cancellation_or_accept_offer
from src.agents.persona_agent import apply_brand_voice

def supervisor_logic(user_input, conversation_state):
    """
    1. Update conversation_state with user_input.
    2. Possibly call mood_agent, reason_agent, offer_agent, or cancel_agent
       depending on conversation state (did we get the reason? has an offer been made? etc.)
    3. Wrap final text with persona_agent.
    4. Return the user-facing message.
    """
    # TODO: Check what's missing in conversation_state
    #  e.g. do we have a reason_for_cancellation? 
    #       has the user insisted on cancel yet?
    
    # Example pseudo-code:
    mood_info = determine_mood_and_urgency(conversation_state["history"])
    conversation_state["mood"] = mood_info

    if not conversation_state.get("reason_for_cancellation"):
        reason = find_cancellation_reason(conversation_state["history"])
        conversation_state["reason_for_cancellation"] = reason["reason_for_cancellation"]
        raw_response_text = "Okay, got your reason. Thanks!"
    else:
        # Offer logic example
        if not conversation_state.get("offer_made"):
            offer_info = propose_offer(conversation_state["reason_for_cancellation"], mood_info)
            conversation_state["offer_info"] = offer_info
            conversation_state["offer_made"] = True
            raw_response_text = offer_info["offer_text"] or "Let me see what I can offer you..."
        else:
            # Cancel/finalize
            result = finalize_cancellation_or_accept_offer(conversation_state["offer_info"])
            conversation_state["status"] = result["status"]
            raw_response_text = result["confirmation_text"]

    # Final brand-voice styling
    stylized_response = apply_brand_voice(raw_response_text)
    return stylized_response
