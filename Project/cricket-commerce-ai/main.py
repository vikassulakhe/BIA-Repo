import ollama
import re

MODEL = "qwen3.5:9b"

current_intent = None


# --------------------------------
# Dummy Order Database
# --------------------------------

orders_db = {
    "12345": {
        "customer": "Rahul Sharma",
        "product": "SG English Willow Bat Grade 2",
        "status": "Shipped",
        "delivery_date": "10 Mar 2026"
    },
    "23456": {
        "customer": "Amit Verma",
        "product": "SS Kashmir Willow Bat",
        "status": "Processing",
        "delivery_date": "12 Mar 2026"
    },
    "45678": {
        "customer": "Rohit Singh",
        "product": "MRF Genius Grand Edition",
        "status": "Out for Delivery",
        "delivery_date": "07 Mar 2026"
    }
}


# --------------------------------
# LLM Call
# --------------------------------

def call_llm(prompt):

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a cricket commerce support assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# --------------------------------
# Extract Order ID
# --------------------------------

def extract_order_id(text):

    match = re.search(r"\b\d{5}\b", text)

    if match:
        return match.group()

    return None


# --------------------------------
# Intent Classification
# --------------------------------

def classify_intent(query):

    prompt = f"""
Classify the customer request.

Categories:
Product inquiry
Order status
Refund / Return

Customer message:
{query}

Return ONLY the category name.
"""

    return call_llm(prompt)


# --------------------------------
# Product Specialist Agent
# --------------------------------

def cricket_specialist(query):

    prompt = f"""
You are a cricket equipment expert.

Answer customer questions about cricket bats.

Customer question:
{query}
"""

    return call_llm(prompt)


# --------------------------------
# Order Agent
# --------------------------------

def order_agent(order_id):

    if order_id not in orders_db:
        return "I could not find that order. Please verify your Order ID."

    order = orders_db[order_id]

    return f"""
Issue Summary:
Order status request

What I Found:
Order {order_id} for {order['product']} belongs to {order['customer']}.

Action Taken:
Checked order database.

Next Steps:
Your order status is **{order['status']}**

Expected Delivery:
{order['delivery_date']}
"""


# --------------------------------
# Refund Agent
# --------------------------------

def refund_agent(order_id):

    if order_id not in orders_db:
        return "I could not find that order. Please verify your Order ID."

    order = orders_db[order_id]

    return f"""
Issue Summary:
Refund request

What I Found:
Order {order_id} for {order['product']} belongs to {order['customer']}.

Action Taken:
Refund request initiated.

Next Steps:
A return label will be sent to your email.

Expected Timeline:
Refund processed within 5–7 business days.
"""


# --------------------------------
# Orchestrator
# --------------------------------

def orchestrator(query):

    global current_intent

    order_id = extract_order_id(query)

    if order_id:

        if current_intent == "Order status":
            return order_agent(order_id)

        elif current_intent == "Refund / Return":
            return refund_agent(order_id)

        else:
            return "Please ask about order status or refund first."

    intent = classify_intent(query)

    if intent:
        current_intent = intent

    if "Product inquiry" in intent:
        return cricket_specialist(query)

    elif "Order status" in intent:
        return "Please provide your 5-digit Order ID."

    elif "Refund" in intent:
        return "Please provide your 5-digit Order ID to start refund."

    else:
        return "I can help with cricket bats, orders, or refunds."