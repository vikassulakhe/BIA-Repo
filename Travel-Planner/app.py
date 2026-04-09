import os
import json
import streamlit as st
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq

# Load .env if present
load_dotenv()

def get_groq_client(provided_key=None):
    api_key = provided_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Set it in environment or enter in sidebar.")
        st.stop()
    return Groq(api_key=api_key)


def search_destination(client, destination):
    prompt = f"""
You are a travel expert. Provide a concise overview about the destination "{destination}" including: top attractions, best time to visit, local tips, safety notes, and quick travel budget tiers (budget/standard/luxury).
Return the answer as a short, well formatted text.
"""

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=400,
    )
    return resp.choices[0].message.content


def generate_itinerary(client, destination, days, interests=""):
    prompt = f"""
You are an expert travel planner. Create a day-by-day itinerary for {days} day(s) in {destination}.
Consider these interests: {interests}
For each day, provide: morning/afternoon/evening suggestions, estimated durations, and short travel tips.
Keep the plan practical and time-aware. Return in plain text.
"""

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=800,
    )
    return resp.choices[0].message.content


def find_flights(client, origin, dest, depart_date, return_date=None):
    prompt = f"""
You are a flight-finder assistant (note: this is a mock search). Provide 3 plausible flight options from {origin} to {dest} departing {depart_date} returning {return_date}.
For each option give: airline, depart/arrive times, duration, stops, and an estimated price in USD.
Return as a short numbered list.
"""

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=400,
    )
    return resp.choices[0].message.content


def find_hotels(client, destination, checkin, checkout):
    prompt = f"""
You are a hotel recommender (mock). Suggest 3 hotel options in {destination} between {checkin} and {checkout}.
For each hotel provide: name, neighbourhood, star rating, short pros/cons, and estimated nightly price in USD.
Return as a numbered list.
"""

    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        max_tokens=400,
    )
    return resp.choices[0].message.content


def parse_numbered_list(text: str):
    """Try to split a numbered list (1., 2., 3.) into items. Fallback to lines."""
    if not text:
        return []
    items = []
    # Split on numbered bullets like '1.' at line starts
    import re
    parts = re.split(r"\n\s*\d+[\.|)]\s+", "\n" + text)
    for p in parts:
        p = p.strip()
        if p:
            items.append(p)
    # If split failed, fallback to paragraphs
    if len(items) <= 1:
        items = [line.strip() for line in text.split("\n") if line.strip()]
    return items


def parse_price(text: str):
    import re
    m = re.findall(r"\$\s?[0-9,]+(?:\.[0-9]{1,2})?", text)
    if m:
        return m[-1]
    m2 = re.findall(r"[0-9,]+\s?USD", text, flags=re.IGNORECASE)
    if m2:
        return m2[-1]
    return None


def parse_flight_details(item_text: str):
    """Heuristic parse of a flight option into structured fields."""
    import re
    lines = [l.strip() for l in item_text.splitlines() if l.strip()]
    title = lines[0] if lines else item_text[:80]
    price = parse_price(item_text) or "N/A"
    # find times like 07:30 or 7:30 AM
    times = re.findall(r"\d{1,2}:\d{2}(?:\s?[APMapm]{2})?", item_text)
    duration = None
    for l in lines:
        if 'duration' in l.lower() or 'hr' in l.lower() or 'h ' in l.lower():
            duration = l
            break
    stops = 'nonstop' if 'nonstop' in item_text.lower() else ("stops" if 'stop' in item_text.lower() or 'layover' in item_text.lower() else "unknown")
    # Attempt to extract airline name (look for common suffixes)
    airline = None
    m = re.search(r"([A-Z][A-Za-z]+(?: Airlines| Airways| Air| AirLines| Airline| Airways)?)", item_text)
    if m:
        airline = m.group(1)
    # Attempt to extract IATA codes (3-letter) for airports
    iatas = re.findall(r"\b[A-Z]{3}\b", item_text)
    return {"title": title, "price": price, "times": times, "duration": duration, "stops": stops, "airline": airline, "iatas": iatas, "raw": item_text}


def parse_hotel_details(item_text: str):
    import re
    lines = [l.strip() for l in item_text.splitlines() if l.strip()]
    title = lines[0] if lines else item_text[:80]
    price = parse_price(item_text) or "N/A"
    # star rating
    stars = None
    m = re.search(r"(\d(\.\d)?)-?star|\b(\d) star\b", item_text, flags=re.IGNORECASE)
    if m:
        stars = m.group(0)
    return {"title": title, "price": price, "stars": stars, "raw": item_text}


def book_item(kind, details):
    # Simple local booking simulation: store in session state
    if "bookings" not in st.session_state:
        st.session_state.bookings = []
    st.session_state.bookings.append({"kind": kind, "details": details, "timestamp": datetime.now(timezone.utc).isoformat()})
    st.success(f"{kind.title()} booked (simulated)")


def main():
    st.set_page_config(page_title="Travel Planner", layout="wide")
    st.title("✈️ Travel Planner (Groq + Streamlit)")

    # Simple custom styling for a cleaner UI
    st.markdown(
        """
    <style>
        :root {
            --bg1: #fff7f0;
            --bg2: #f0f9ff;
            --accent1: #e64a4a; /* slightly darker red */
            --accent2: #3b82f6; /* stronger blue */
            --text: #0b1220; /* main text color - dark */
            --muted: #475569; /* muted/dim text */
        }
        html, body, .viewerRoot {
            background: radial-gradient(circle at 10% 10%, var(--bg1), var(--bg2));
            color: var(--text);
        }
        header, .stApp, .main { background: transparent; color: var(--text); }
        .title {font-size:32px; font-weight:800; color:var(--accent2); text-shadow:0 2px 6px rgba(0,0,0,0.06);} 
        /* Ensure buttons stay readable */
        .stButton>button {
            background: linear-gradient(90deg, var(--accent1), var(--accent2));
            color: #ffffff; border: none; padding: 8px 12px; border-radius: 8px;
            box-shadow: 0 6px 18px rgba(59,130,246,0.12);
        }
        .stButton>button:hover { filter: brightness(1.02); }
        .card { border-radius:12px; padding:14px; background: linear-gradient(180deg, #ffffff, #f6fbff); border: 1px solid rgba(59,130,246,0.08); box-shadow: 0 6px 20px rgba(59,130,246,0.05); color: var(--text); }
        .card-header { font-weight:700; color:var(--text); margin-bottom:8px; font-size:16px; }
        .muted { color:var(--muted); font-size:13px; }
        .action{display:flex; gap:8px; align-items:center}
        .stSidebar { background: linear-gradient(180deg, #ffffffcc, #f7fbffcc); border-radius:8px; padding:8px; color:var(--text); }
        /* Make form labels and inputs darker */
        label, .css-1x0xodm, .stTextInput>div>input, .stDateInput, .stNumberInput { color: var(--text); }
    </style>
            """,
            unsafe_allow_html=True,
        )

    # Sidebar: API key and user profile
    st.sidebar.header("Settings")
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        st.sidebar.info("Using GROQ_API_KEY from environment")
    provided_key = st.sidebar.text_input("Groq API Key (optional)", type="password", value=env_key or "")

    client = None
    try:
        client = get_groq_client(provided_key=provided_key)
    except Exception:
        return

    # Tabs: Planner and Bookings
    tab1, tab2 = st.tabs(["Planner", "Bookings"])

    with tab1:
        # Inputs
        col1, col2 = st.columns([2, 1])

        with col1:
            destination = st.text_input("Destination (city or country)", value="Paris")
            origin = st.text_input("Origin city/airport", value="New York")
            dates = st.date_input("Departure date", value=datetime.now(timezone.utc).date())
            return_date = st.date_input("Return date (optional)", value=None)
            days = st.number_input("Number of days (itinerary)", min_value=1, max_value=30, value=3)
            interests = st.text_input("Interests (comma separated, e.g. museums, food, hiking)")

        if st.button("Search Destination"):
            with st.spinner("Searching destination..."):
                try:
                    info = search_destination(client, destination)
                    st.subheader(f"About {destination}")
                    st.write(info)
                except Exception as e:
                    st.error(f"Destination search failed: {e}")

        if st.button("Generate Itinerary"):
            with st.spinner("Generating itinerary..."):
                try:
                    plan = generate_itinerary(client, destination, days, interests)
                    st.subheader(f"{days}-day Itinerary for {destination}")
                    st.write(plan)
                except Exception as e:
                    st.error(f"Itinerary generation failed: {e}")

    with col2:
        st.subheader("Flights")
        if st.button("Find Flights"):
            with st.spinner("Searching flights (mock)..."):
                try:
                    flights = find_flights(client, origin, destination, dates.isoformat(), return_date.isoformat() if return_date else "N/A")
                    options = parse_numbered_list(flights)
                    st.session_state['flight_text'] = flights
                    st.session_state['flight_options'] = options
                except Exception as e:
                    st.error(f"Flight search failed: {e}")

        # Render flight options as cards for nicer UI
        flight_options = st.session_state.get('flight_options')
        if flight_options:
            for idx, opt in enumerate(flight_options):
                with st.container():
                    st.markdown(f"<div class='card'><div class='card-header'>{opt.split('\n')[0]}</div>\n<div class='muted'>{opt.split('\n',1)[-1]}</div></div>", unsafe_allow_html=True)
                    col_a, col_b = st.columns([3,1])
                    with col_b:
                        if st.button("Book Flight", key=f"book_flight_{idx}"):
                            parsed = parse_flight_details(opt)
                            book_item("flight", parsed)
        elif st.session_state.get('flight_text'):
            st.write(st.session_state.get('flight_text'))

        st.subheader("Hotels")
        if st.button("Find Hotels"):
            with st.spinner("Searching hotels (mock)..."):
                try:
                    hotels = find_hotels(client, destination, dates.isoformat(), (return_date.isoformat() if return_date else (dates.isoformat())))
                    options = parse_numbered_list(hotels)
                    st.session_state['hotel_text'] = hotels
                    st.session_state['hotel_options'] = options
                except Exception as e:
                    st.error(f"Hotel search failed: {e}")

        hotel_options = st.session_state.get('hotel_options')
        if hotel_options:
            for idx, opt in enumerate(hotel_options):
                with st.container():
                    st.markdown(f"<div class='card'><div class='card-header'>{opt.split('\n')[0]}</div>\n<div class='muted'>{opt.split('\n',1)[-1]}</div></div>", unsafe_allow_html=True)
                    col_a, col_b = st.columns([3,1])
                    with col_b:
                        if st.button("Book Hotel", key=f"book_hotel_{idx}"):
                            parsed_h = parse_hotel_details(opt)
                            book_item("hotel", parsed_h)
        elif st.session_state.get('hotel_text'):
            st.write(st.session_state.get('hotel_text'))

    # Bookings tab content
    with tab2:
        st.header("Your Bookings (simulated)")
        if "bookings" in st.session_state and st.session_state.bookings:
            for i, b in list(enumerate(st.session_state.bookings)):
                title = b['details'].get('title') if isinstance(b['details'], dict) else str(b['details'])[:40]
                with st.expander(f"{i+1}. {b['kind'].title()} - {title}"):
                    # show structured details if available
                    if isinstance(b['details'], dict):
                        st.json(b['details'])
                    else:
                        st.write(b['details'])
                    st.write("Booked at:", b['timestamp'])
                    # download booking JSON
                    st.download_button(
                        label="Export booking (JSON)",
                        data=json.dumps(b, indent=2),
                        file_name=f"booking_{i+1}.json",
                        mime="application/json",
                    )
                    if st.button(f"Cancel booking #{i+1}", key=f"cancel_{i}"):
                        st.session_state.bookings.pop(i)
                        st.success("Booking cancelled")
                        st.experimental_rerun()
        else:
            st.info("No bookings yet")
    st.sidebar.markdown("---")
    st.sidebar.markdown("This app simulates booking flows and uses Groq to generate destination info and itineraries. For real bookings integrate airline/hotel APIs.")


if __name__ == "__main__":
    main()
