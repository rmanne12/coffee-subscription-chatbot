import json
from src.agents.mood_agent import determine_mood_and_urgency
from src.agents.reason_agent import find_cancellation_reason
from src.agents.offer_agent import propose_offer
from src.agents.decision_agent import finalize_cancellation_or_accept_offer
from src.agents.persona_agent import apply_brand_voice

PRODUCTS = {
    "decaf": {
        "name": "Decaf: Live Long and De-Caffeinate",
        "caffeine": 10,
        "flavor": "hazelnut"
    },
    "half-caf": {
        "name": "Half-Caf: One Small Sip for Man",
        "caffeine": 50,
        "flavor": "vanilla"
    },
    "regular": {
        "name": "Regular Caf: Where No Bean Has Been Before",
        "caffeine": 100,
        "flavor": "regular"
    },
    "super-caf": {
        "name": "Super Caf: Warp Speed Blend",
        "caffeine": 200,
        "flavor": "regular"
    }
}

STAGE_ASK_REASON = "ask_reason"
STAGE_ASK_PLAN = "ask_plan"
STAGE_DISCUSS_REASON = "discuss_reason"
STAGE_MAKE_OFFER = "make_offer"
STAGE_FINALIZE = "finalize"


def supervisor_logic(user_input, conversation_state):
    """
    Extended logic to handle:
    1. Multiple reasons or changed reasons (e.g. "It's too expensive" => "Actually I don't like the taste")
    2. Adapts tone if user is very angry
    3. Allows multiple offers
    4. Recommends new products for taste/caffeine
    5. Uses Persona Agent last, applying space-themed brand voice
    """

    conversation_state["history"].append({"user": user_input})

    mood_info = determine_mood_and_urgency(conversation_state["history"])
    conversation_state["mood"] = mood_info 

    keywords = ["taste", "flavor", "jittery", "moving", "expensive", "stockpiled"]
    if any(kw in user_input.lower() for kw in keywords):
        new_reason_data = find_cancellation_reason(conversation_state["history"])
        new_reason = new_reason_data["reason_for_cancellation"]
        if new_reason and new_reason != conversation_state.get("reason_for_cancellation"):
            conversation_state["reason_for_cancellation"] = new_reason
            if conversation_state.get("stage") != STAGE_MAKE_OFFER:
                conversation_state["stage"] = STAGE_DISCUSS_REASON

    if (
        "cancel" in user_input.lower()
        and mood_info["mood"] in ["angry", "frustrated"]
        and mood_info["urgency"] == "high"
    ):
        result = {
            "status": "cancelled",
            "confirmation_text": (
                "Your subscription is now canceled. "
                "We appreciate your time with us and hope to serve you again."
            )
        }
        conversation_state["status"] = result["status"]
        final_text = apply_brand_voice(
            result["confirmation_text"],
            mood=mood_info["mood"],
            subscription_status=result["status"]
        )
        conversation_state["history"].append({"bot": final_text})
        _debug_print(conversation_state)
        return final_text

    stage = conversation_state.get("stage")
    if not stage:
        stage = STAGE_ASK_REASON
        conversation_state["stage"] = stage

    reason_for_cancellation = conversation_state.get("reason_for_cancellation")
    user_coffee_plan = conversation_state.get("user_coffee_plan")  

    if "offer_info" not in conversation_state:
        conversation_state["offer_info"] = {"offers": []}
    offer_info = conversation_state["offer_info"]

    bot_reply = ""

    # STAGE 1: ASK_REASON
    if stage == STAGE_ASK_REASON and not reason_for_cancellation:
        reason_data = find_cancellation_reason(conversation_state["history"])
        reason_detected = reason_data["reason_for_cancellation"]

        if reason_detected:
            conversation_state["reason_for_cancellation"] = reason_detected
            conversation_state["stage"] = STAGE_ASK_PLAN
            bot_reply = (
                f"Thanks for letting me know you want to cancel because it's '{reason_detected}'. "
                "Which coffee blend are you currently subscribed to (Decaf, Half-Caf, Regular, or Super Caf)?"
            )
        else:
            bot_reply = (
                "I want to help you out! Could you tell me specifically why you’re canceling?"
            )

    # STAGE 2: ASK_PLAN
    elif stage == STAGE_ASK_PLAN and not user_coffee_plan:
        found_plan = _extract_coffee_plan(user_input)
        if found_plan:
            conversation_state["user_coffee_plan"] = found_plan
            conversation_state["stage"] = STAGE_DISCUSS_REASON
            bot_reply = (
                f"Got it, you’re on {PRODUCTS[found_plan]['name']}. "
                "Let’s talk more about what’s prompting your cancellation. Could you share more details?"
            )
        else:
            bot_reply = (
                "Which coffee plan do you have right now? Decaf, Half-Caf, Regular, or Super Caf?"
            )

    # STAGE 3: DISCUSS_REASON
    elif stage == STAGE_DISCUSS_REASON and reason_for_cancellation and user_coffee_plan:
        reason = reason_for_cancellation.lower()
        plan = user_coffee_plan.lower()
        followup_response = _reason_followup(reason, plan)

        conversation_state["stage"] = STAGE_MAKE_OFFER
        bot_reply = followup_response

    # STAGE 4: MAKE_OFFER (No offer made yet)
    elif (
        stage == STAGE_MAKE_OFFER
        and reason_for_cancellation
        and user_coffee_plan
        and not conversation_state.get("offer_made")
    ):
        from_offer_agent = propose_offer(reason_for_cancellation, mood_info)
        offer_info["offers"].append(from_offer_agent)

        conversation_state["offer_info"] = offer_info
        conversation_state["offer_made"] = True
        bot_reply = (
            f"I have an idea: {from_offer_agent['offer_text']}\n"
            "Does that sound like it could help?"
        )

    # STAGE 4 (continued): MAKE_OFFER
    elif stage == STAGE_MAKE_OFFER and conversation_state.get("offer_made"):
        finalization = finalize_cancellation_or_accept_offer(conversation_state["history"], offer_info)
        conversation_state["status"] = finalization["status"]

        if finalization.get("offer_accepted"):
            if conversation_state["offer_info"]["offers"]:
                conversation_state["offer_info"]["offers"][-1]["offer_accepted"] = True
            bot_reply = finalization["confirmation_text"]
        else:
            if "cancel" in user_input.lower():
                bot_reply = finalization["confirmation_text"]
            else:
                second_offer = {
                    "offer_type": "pause up to 6 months",
                    "offer_text": "We can also pause your subscription for up to 6 months if that’s better!",
                    "offer_accepted": None
                }
                offer_info["offers"].append(second_offer)
                conversation_state["offer_info"] = offer_info
                bot_reply = (
                    f"I understand—how about another option? {second_offer['offer_text']} "
                    "Would you like to try that?"
                )

    # STAGE 5: FINALIZE OR FALLBACK
    else:
        if conversation_state.get("status") == "cancelled":
            bot_reply = "Your subscription is already cancelled. Thanks for being with us!"
        elif conversation_state.get("status") == "active":
            bot_reply = "Your subscription remains active—anything else I can assist with?"
        else:
            bot_reply = "Let me know if there's anything else I can do for you, Captain."

    mood = mood_info.get("mood", "")
    if mood in ["angry", "frustrated"] and mood_info.get("urgency") == "high":
        toned_down_reply = "I understand your frustration. " + bot_reply.replace("🚀", "").replace("✨", "")
        bot_reply = toned_down_reply

    final_text = apply_brand_voice(
        bot_reply,
        mood=mood,
        reason_for_cancellation=conversation_state.get("reason_for_cancellation"),
        subscription_status=conversation_state.get("status"),
        additional_context="User coffee plan: " + str(conversation_state.get("user_coffee_plan"))
    )

    conversation_state["history"].append({"bot": final_text})
    _debug_print(conversation_state)

    return final_text


def _debug_print(conversation_state):
    debug_info = {
        "mood": conversation_state.get("mood"),
        "stage": conversation_state.get("stage"),
        "reason_for_cancellation": conversation_state.get("reason_for_cancellation"),
        "user_coffee_plan": conversation_state.get("user_coffee_plan"),
        "offer_info": conversation_state.get("offer_info"),
        "status": conversation_state.get("status")
    }
    print("\n[DEBUG] Current State:")
    print(json.dumps(debug_info, indent=2))
    print("[DEBUG] End of State\n")


def _extract_coffee_plan(user_message):
    """
    Returns a normalized key from PRODUCTS if user mentions 'decaf', 'half', 'regular', 'super'
    """
    user_lower = user_message.lower()
    if "decaf" in user_lower:
        return "decaf"
    elif "half" in user_lower:
        return "half-caf"
    elif "regular" in user_lower:
        return "regular"
    elif "super" in user_lower:
        return "super-caf"
    return None


def _reason_followup(reason, plan):
    """
    Example bridging step to gather more info or confirm details
    before making an offer. Adapt as needed:
    """
    if reason == "too expensive":
        return (
            "I hear you. Balancing budgets can be tough. Let’s see if we can lighten your load. "
            "Would you like to see how we might help with cost or scheduling?"
        )
    elif reason == "taste":
        return (
            f"Understood. You mentioned taste issues with {PRODUCTS[plan]['name']}. "
            "Is it the flavor, the caffeine level, or something else?"
        )
    elif reason == "stockpiled":
        return (
            "Got it! Sounds like you might have enough coffee for a small moon colony. "
            "Let’s see if we can help pause or skip deliveries?"
        )
    elif reason == "too jittery":
        return (
            "If it’s too much caffeine, we could suggest a lighter option. "
            "Let’s discuss how to keep you from going into hyperdrive!"
        )
    elif reason == "not jittery enough":
        return (
            "So you need a bigger energy boost? We might have something stronger! "
            "Let’s figure out a warp-speed solution for you."
        )
    elif reason == "moving":
        return (
            "Moving can be stressful. Would pausing or skipping your deliveries help "
            "while you’re in transit?"
        )
    else:
        return (
            "I see. Let’s figure out if there’s a solution that works for your situation."
        )