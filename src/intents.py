"""
intents.py
Single source of truth for the customer support intent taxonomy.
"""

INTENTS = {
    "account_access": "login, password, verification, account locked/suspended issues",
    "account_settings": "unsubscribe requests, notification/email preferences",
    "agent_escalation": "complaints about being ignored, poor service, wants a human, repeated contact attempts failed",
    "contact_request": "asking for an email/phone/contact method",
    "delivery_delay": "package late, not yet arrived, stuck in transit",
    "order_status": "asking where an order is, order number references, general order inquiries",
    "product_availability": "asking if a product is sold/available in a region",
    "wrong_delivery": "received wrong item or unordered package",
    "other": "doesn't fit any category above"
}
