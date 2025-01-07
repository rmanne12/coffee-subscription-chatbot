import json
from langchain.chains.base import Chain
from typing import Dict, Any

from langchain_src.chains.mood_chain import build_mood_chain, parse_mood_output
from langchain_src.chains.reason_chain import build_reason_chain, parse_reason_output, VALID_REASONS
from langchain_src.chains.offer_chain import build_offer_chain, parse_offer_output
from langchain_src.chains.decision_chain import build_decision_chain, parse_decision_output
from langchain_src.chains.persona_chain import build_persona_chain

class UserFlowChain(Chain):
    """
    A single chain that orchestrates:
      1) mood detection
      2) reason detection
      3) propose offer
      4) finalize decision
      5) persona rewrite
    """

    conversation_key: str = "conversation"
    output_key: str = "final_response"

    @property
    def input_keys(self):
        return [self.conversation_key]

    @property
    def output_keys(self):
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        conversation_text = inputs[self.conversation_key]
        stage = inputs.get("stage", "initial")  # Default to initial stage

        try:
            if stage == "initial":
                # Step 1: Mood Detection
                mood_chain = build_mood_chain()
                mood_json = mood_chain.run({"conversation": conversation_text})
                # print(f"DEBUG: Mood Chain Output = {mood_json}")  # Debugging
                mood_data = parse_mood_output(mood_json)
                # print(f"DEBUG: Parsed Mood Data = {mood_data}")  # Debugging
                mood = mood_data.get("mood", "neutral")
                urgency = mood_data.get("urgency", "low")

                # Step 2: Reason Detection
                reason_chain = build_reason_chain()
                reason_json = reason_chain.run({
                    "conversation": conversation_text,
                    "valid_reasons": ", ".join(VALID_REASONS)
                })
                # print(f"DEBUG: Reason Chain Output = {reason_json}")  # Debugging
                reason_value = parse_reason_output(reason_json) or "unknown"

                # Return combined mood and reason for further processing
                return {
                    "mood": mood,
                    "urgency": urgency,
                    "reason": reason_value,
                    "final_response": (
                        f"We detected that your mood is '{mood}' and your reason for cancellation is '{reason_value}'. "
                        "Let’s proceed to find the best solution for you."
                    )
                }

            elif stage == "offer":
                # Step 3: Offer Proposal
                offer_chain = build_offer_chain()
                offer_json = offer_chain.run({
                    "reason": inputs["reason"],
                    "mood": inputs["mood"],
                    "urgency": inputs["urgency"],
                    "existing_offers": "[]"
                })
                # print(f"DEBUG: Offer Chain Output = {offer_json}")  # Debugging
                offer_data = parse_offer_output(offer_json)
                # print(f"DEBUG: Parsed Offer Data = {offer_data}")  # Debugging

                return {"offer": offer_data, "final_response": offer_data["offer_text"]}

            elif stage == "confirmation":
                # Step 4: Confirmation
                decision_chain = build_decision_chain()
                decision_json = decision_chain.run({
                    "conversation": conversation_text,
                    "offers_so_far": json.dumps([inputs.get("offer", {})], indent=2)
                })
                # print(f"DEBUG: Decision Chain Output = {decision_json}")  # Debugging
                decision_data = parse_decision_output(decision_json)
                # print(f"DEBUG: Parsed Decision Data = {decision_data}")  # Debugging

                return {
                    "final_response": decision_data["confirmation_text"]
                }

        except Exception as e:
            print(f"DEBUG: Error in _call = {e}")
            raise


    def _chain_type(self) -> str:
        return "custom"

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Overrides the base Chain `run` method to ensure the correct output format.
        """
        result = self._call(inputs)
        if not isinstance(result, dict):
            raise TypeError(f"Expected dictionary output but got {type(result).__name__}: {result}")
        if self.output_key not in result:
            raise KeyError(f"Missing output key: {self.output_key}")
        return result


def create_userflow_chain():
    return UserFlowChain()
