def finalize_cancellation_or_accept_offer(offer_info):
    """
    If offer_info indicates user accepted or declined,
    finalize the subscription status.
    Return a dict: {
      "status": "cancelled" or "active",
      "confirmation_text": "Your subscription is now cancelled." (or something else)
    }
    """
    # TODO: Implement logic based on conversation or user acceptance
    return {
        "status": "cancelled",  # or "active"
        "confirmation_text": "Subscription cancelled. Sorry to see you go!"
    }
