# streamlit_app.py (Final Version with Horizontal Icon Buttons)
import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path

from src.config import QUERIES_DIR

# --- Page Configuration ---
st.set_page_config(
    page_title="Fake News Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# --- Helper Functions (no changes here) ---
def get_queries():
    try:
        response = requests.get(f"{API_URL}/queries")
        response.raise_for_status()
        return response.json().get("queries", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API: {e}")
        return None

def move_to_trash(query_id):
    try:
        response = requests.delete(f"{API_URL}/trash/{query_id}")
        response.raise_for_status()
        st.toast(f"Query '{query_id}' moved to trash.", icon="🗑️")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to move to trash: {e.response.json().get('detail')}")
        return False

def rerun_query(query_id):
    try:
        response = requests.post(f"{API_URL}/rerun/{query_id}")
        response.raise_for_status()
        st.toast(f"✅ Successfully queued '{query_id}' for rerun!", icon="🔄")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to rerun query: {e.response.json().get('detail')}")
        return False

# --- Main Page UI ---
st.title("🛡️ Fake News Detection Dashboard")
st.markdown("Monitor the status of all news queries and view their analysis reports.")

if 'search_term' not in st.session_state:
    st.session_state.search_term = ""

search_term = st.text_input(
    "🔎 Search Queries",
    value=st.session_state.search_term,
    placeholder="Type to search by Query ID or Caption content...",
    help="Your query list will update in real-time as you type."
)
st.session_state.search_term = search_term

if st.button("Refresh", key="main_refresh"):
    st.session_state.search_term = ""
    st.rerun()

queries = get_queries()

if queries is None:
    st.warning("Could not load data from API. Is the API server running?")
elif not queries:
    st.info("No queries found. Add a new query using the 'Add New Query' page.")
else:
    # --- Filter logic (no changes here) ---
    filtered_queries = []
    search_lower = st.session_state.search_term.lower()
    for q in queries:
        caption_text = "[Caption not found]"
        try:
            caption_path = next((QUERIES_DIR / q['query_id']).glob("*.txt"))
            caption_text = caption_path.read_text().strip()
        except (StopIteration, FileNotFoundError):
            pass
        q['caption_text'] = caption_text
        if search_lower in q['query_id'].lower() or search_lower in q['caption_text'].lower():
            filtered_queries.append(q)

    st.subheader(f"Displaying {len(filtered_queries)} of {len(queries)} Queries")
    st.divider()

    if not filtered_queries:
        st.warning("No queries match your search term.")
    else:
        for q in sorted(filtered_queries, key=lambda x: x['created_at'], reverse=True):
            if q['status'] == 'trashed': continue
            
            query_id = q['query_id']
            status = q['status']
            stages = json.loads(q['stages'])
            status_icon = {"pending": "🕒", "processing": "⚙️", "completed": "✅", "failed": "❌"}
            verdict = q.get('verdict', 'N/A')
            verdict_icon = {"True": "✅", "Fake": "❌", "Uncertain": "❓", "Error": "🔥"}
            caption_text = q['caption_text']

            with st.container(border=True):
                col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
                with col1:
                    st.markdown(f"**Query ID:** `{query_id}`")
                    st.info(f"**Caption:** {caption_text[:100] + ('...' if len(caption_text) > 100 else '')}")
                with col2:
                    st.markdown(f"**Status:** {status_icon.get(status, '❓')} {status.title()}")
                    st.markdown(f"""<small>{status_icon.get(stages['evidence_extraction'])} Evidence<br>{status_icon.get(stages['model_inference'])} Inference<br>{status_icon.get(stages['pdf_generation'])} PDF</small>""", unsafe_allow_html=True)
                with col3:
                    st.markdown("**Verdict**")
                    if status == 'completed':
                        if verdict == "True": st.success(f"{verdict_icon[verdict]} {verdict}", icon=verdict_icon[verdict])
                        elif verdict == "Fake": st.error(f"{verdict_icon[verdict]} {verdict}", icon=verdict_icon[verdict])
                        else: st.warning(f"{verdict_icon.get(verdict, '❓')} {verdict}", icon=verdict_icon.get(verdict, '❓'))
                    else: st.caption("Processing...")
                    if status == 'failed':
                        with st.expander("Show Error"): st.error(f"{q.get('error_message', 'Unknown error')}")
                
                with col4:
                    st.markdown("**Actions**")
                    
                    # --- NEW: Horizontal Icon Button Layout ---
                    # Create sub-columns for the action buttons
                    b_col1, b_col2, b_col3, b_col4 = st.columns([1, 1, 1, 1])

                    with b_col1:
                        if st.button("📄", key=f"details_{query_id}", help="View analysis details page"):
                            st.session_state.selected_query_id = query_id
                            st.switch_page("pages/2_Query_Details.py")
                    
                    with b_col2:
                        # For the PDF, st.link_button is the best choice. We'll style it to look similar.
                        if status == 'completed' and q.get("result_pdf_path"):
                            st.link_button("📜", f"{API_URL}/results/{query_id}", help="Open final PDF report")

                    with b_col3:
                        if st.button("🔄", key=f"rerun_{query_id}", help="Rerun analysis for this query"):
                            rerun_query(query_id); st.rerun()

                    with b_col4:
                        if st.button("🗑️", key=f"trash_{query_id}", help="Move this query to the trash"):
                            if move_to_trash(query_id): st.rerun()