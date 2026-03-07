from langchain_community.llms import Ollama
import re

current_intent = None

# -----------------------------
# LLM Setup
# -----------------------------

llm = Ollama(model="llama3")

# -----------------------------
# Fake Order Database
# -----------------------------

orders = {
    "12345": {
        "product": "SG English Willow Bat Grade 2",
        "customer": "Rahul Sharma",
        "status": "Delivered",
        "delivery": "Delivered on 02 Mar 2026"
    }
}

# -----------------------------
# Helper Function
# -----------------------------

def extract_order_id(text):
    match = re.search(r"\b\d{5}\b", text)
    return match.group() if match else None

# -----------------------------
# Intent Classifier Agent
# -----------------------------

def classify_intent(user_input):

    prompt = f"""
Classify the intent of this message into ONE category.

Product inquiry
Order status
Refund / Return
Damaged bat
Warranty claim
Payment issue
Delivery problem
Complaint / escalation
Fraud concern
Other

Message: {user_input}

Return ONLY the category name.
"""

    response = llm.invoke(prompt).strip()

    return response

# -----------------------------
# Product Expert Agent
# -----------------------------

def product_agent(query):

    prompt = f"""
You are a professional cricket bat expert.

Answer questions about cricket bats only.

Topics include:
- English willow vs Kashmir willow
- Bat grades
- Bat weight selection
- Sweet spot
- Knock-in process
- Bat care

Customer question:
{query}
"""

    return llm.invoke(prompt)

# -----------------------------
# Order Agent
# -----------------------------

def order_agent(message):

    order_id = extract_order_id(message)

    if not order_id:
        return "Please provide your 5-digit Order ID so I can check your cricket bat order."

    if order_id not in orders:
        return "Order ID not found. Please verify your order number."

    order = orders[order_id]

    return f"""
Issue Summary:
Customer requested order status.

What I Found:
Order {order_id} for {order['product']} belongs to {order['customer']}.

Action Taken:
Checked order database.

Next Steps:
The order is currently {order['status']}.

Expected Timeline:
{order['delivery']}
"""

# -----------------------------
# Refund Agent
# -----------------------------

def refund_agent(message):

    order_id = extract_order_id(message)

    if not order_id:
        return "Please provide your 5-digit Order ID so I can process the refund request."

    if order_id not in orders:
        return "Order ID not found. Please verify the order number."

    order = orders[order_id]

    prompt = f"""
Customer requested refund for a cricket bat order.

Order Details:
Order ID: {order_id}
Product: {order['product']}
Status: {order['status']}

Important Rules:
- Do NOT approve refund automatically
- Refund requires inspection
- Only register the refund request

Respond using this format:

Issue Summary:
What I Found:
Action Taken:
Next Steps:
Expected Timeline:

Refund will be processed only after return inspection.
"""

    return llm.invoke(prompt)

# -----------------------------
# Warranty Agent
# -----------------------------

def warranty_agent(message):

    order_id = extract_order_id(message)

    if not order_id:
        return "Please provide your 5-digit Order ID so I can check the warranty details for your cricket bat."

    if order_id not in orders:
        return "Order ID not found. Please verify the order number."

    order = orders[order_id]

    prompt = f"""
Customer reported a damaged cricket bat and requested warranty support.

Order Details:
Order ID: {order_id}
Product: {order['product']}
Status: {order['status']}

Rules:
- Do NOT approve warranty automatically
- Warranty requires inspection
- Respond only for cricket bats

Use this format:

Issue Summary:
What I Found:
Action Taken:
Next Steps:
Expected Timeline:
"""

    return llm.invoke(prompt)

# -----------------------------
# Orchestrator Agent
# -----------------------------

def orchestrator(user_input):

    global current_intent

    order_id = extract_order_id(user_input)

    # If user sends order id, continue previous workflow
    if order_id and current_intent == "Order status":
        return order_agent(user_input)

    if order_id and current_intent == "Refund / Return":
        return refund_agent(user_input)

    if order_id and current_intent == "Warranty claim":
        return warranty_agent(user_input)

    # Otherwise classify new intent
    intent = classify_intent(user_input)

    current_intent = intent

    print("\nDetected Intent:", intent)

    if "Product" in intent:
        return product_agent(user_input)

    elif "Order status" in intent:
        return order_agent(user_input)

    elif "Refund" in intent or "Return" in intent:
        return refund_agent(user_input)

    elif "Warranty" in intent or "Damaged bat" in intent:
        return warranty_agent(user_input)

    else:
        return "Sorry, this request is outside my scope. I can only assist with cricket bat related queries."
    

# -----------------------------
# Chat Loop
# -----------------------------

def chat():

    print("\n🏏 Cricket Bat Support Multi-Agent System")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("Customer: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        response = orchestrator(user_input)

        print("\nAgent:", response)

# -----------------------------
# Run Program
# -----------------------------

if __name__ == "__main__":
    chat()