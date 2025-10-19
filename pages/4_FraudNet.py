# pages/5_FraudNet_Results.py
import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path

from src.config import QUERIES_DIR

# --- Page Configuration ---
st.set_page_config(
    page_title="FraudNet Results",
    page_icon="🔎",
    layout="wide"
)

API_URL = "https://maida-prayerless-brambly.ngrok-free.dev"

# --- Helper Functions (reused from dashboard) ---
def get_queries():
    try:
        response = requests.get(f"{API_URL}/queries")
        response.raise_for_status()
        return response.json().get("queries", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API: {e}")
        return None

# --- Main Page UI ---
st.title("🔎 FraudNet Model Results")
st.markdown("This page displays the specific predictions made by the FraudNet model for each completed query.")

# --- Search Bar (reused from dashboard) ---
if 'fraudnet_search_term' not in st.session_state:
    st.session_state.fraudnet_search_term = ""

search_term = st.text_input(
    "Search Queries",
    value=st.session_state.fraudnet_search_term,
    placeholder="Type to search by Query ID or Caption content...",
    key="fraudnet_search"
)
st.session_state.fraudnet_search_term = search_term

if st.button("Refresh Data", key="fraudnet_refresh"):
    st.session_state.fraudnet_search_term = ""
    st.rerun()

queries = get_queries()

if queries is None:
    st.warning("Could not load data from API. Is the API server running?")
elif not queries:
    st.info("No queries found.")
else:
    # --- Data Preparation ---
    fraudnet_results = []
    for q in queries:
        # We only care about completed queries that will have a result
        if q['status'] != 'completed':
            continue

        # Extract caption text for searching and display
        caption_text = "[Caption not found]"
        try:
            caption_path = next((QUERIES_DIR / q['query_id']).glob("*.txt"))
            caption_text = caption_path.read_text().strip()
        except (StopIteration, FileNotFoundError):
            pass
        q['caption_text'] = caption_text

        # Extract FraudNet specific results
        verdict = "N/A"
        confidence = 0.0
        try:
            # We need to fetch the full details to get the FraudNet response
            details_response = requests.get(f"{API_URL}/details/{q['query_id']}")
            if details_response.status_code == 200:
                details_data = details_response.json()
                fn_response = details_data.get('results', {}).get('fraudnet_response', {})
                label = fn_response.get('fraudnet_label')
                
                if label == 1:
                    verdict = "Fake"
                    confidence = fn_response.get('confidence', 0.0)
                elif label == 0:
                    verdict = "True"
                    # Invert confidence for "True" as per original logic if needed
                    confidence = fn_response.get('confidence', 0.0)
                else:
                    verdict = "N/A"

                q['fraudnet_verdict'] = verdict
                q['fraudnet_confidence'] = confidence
                fraudnet_results.append(q)

        except requests.exceptions.RequestException:
            # If fetching details fails, skip this query for this view
            continue

    # --- Filtering Logic (reused from dashboard) ---
    filtered_results = []
    search_lower = st.session_state.fraudnet_search_term.lower()

    if not search_lower:
        filtered_results = fraudnet_results
    else:
        for res in fraudnet_results:
            if search_lower in res['query_id'].lower() or search_lower in res['caption_text'].lower():
                filtered_results.append(res)
    
    st.subheader(f"Displaying {len(filtered_results)} FraudNet Predictions")
    st.divider()

    if not filtered_results:
        st.warning("No completed queries match your search term.")
    else:
        # --- Display in a clean, structured list ---
        for res in sorted(filtered_results, key=lambda x: x['created_at'], reverse=True):
            query_id = res['query_id']
            verdict = res['fraudnet_verdict']
            confidence = res['fraudnet_confidence']
            verdict_icon = {"True": "✅", "Fake": "❌"}

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**Query ID:** `{query_id}`")
                    st.info(f"**Caption:** {res['caption_text'][:120] + '...' if len(res['caption_text']) > 120 else res['caption_text']}")

                with col2:
                    st.markdown("**FraudNet Verdict**")
                    if verdict == "True":
                        st.success(f"{verdict}", icon="✅")
                    elif verdict == "Fake":
                        st.error(f"{verdict}", icon="❌")
                    else:
                        st.warning("N/A")

                with col3:
                    st.markdown(f"**Confidence**")
                    st.metric(label="", value=f"{confidence:.2%}")
                    
                with col4:
                    st.markdown("**Actions**")
                    # --- NEW: Details Button ---
                    if st.button("View Details", key=f"fn_details_{query_id}", use_container_width=True):
                        # Set the unique session state key and switch page
                        st.session_state.selected_fraudnet_query_id = query_id
                        st.switch_page("pages/6_FraudNetDetails.py")