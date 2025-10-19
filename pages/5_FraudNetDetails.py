# pages/6_FraudNet_Details.py
import streamlit as st
import requests
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(
    page_title="FraudNet Details",
    page_icon="🔎",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

# --- Main Page Logic ---
# Retrieve the query ID from session state
if "selected_fraudnet_query_id" not in st.session_state:
    st.error("No Query ID specified. Please select a query from the FraudNet Results page.")
    if st.button("Back to FraudNet Results", icon="⬅️"):
        st.switch_page("pages/5_FraudNet_Results.py")
else:
    query_id = st.session_state.selected_fraudnet_query_id
    
    try:
        # Fetch the full details for the selected query
        response = requests.get(f"{API_URL}/details/{query_id}")
        response.raise_for_status()
        data = response.json()

        metadata = data.get('metadata', {})
        fn_response = data.get('results', {}).get('fraudnet_response', {})

        if not metadata or not fn_response:
            st.error("Could not load complete data for this query.")
            st.stop()

        # --- Extract FraudNet Verdict and Confidence ---
        label = fn_response.get('fraudnet_label')
        verdict = "N/A"
        confidence = 0.0
        
        if label == 1:
            verdict = "Fake"
            confidence = fn_response.get('confidence', 0.0)
        elif label == 0:
            verdict = "True"
            confidence = 1.0 - fn_response.get('confidence', 0.0)

        # --- Page Title and Header ---
        st.title(f"🔎 FraudNet Details for Query: `{query_id}`")
        if st.button("Back to FraudNet Results", icon="⬅️"):
            st.switch_page("pages/5_FraudNet_Results.py")
        st.divider()

        # --- Verdict and Confidence Display ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("FraudNet Verdict")
            if verdict == "True":
                st.success(f"## {verdict}", icon="✅")
            elif verdict == "Fake":
                st.error(f"## {verdict}", icon="❌")
            else:
                st.warning("## Uncertain", icon="❓")
        
        with col2:
            st.subheader("Model Confidence")
            st.metric(label="Confidence Score", value=f"{confidence:.2%}")
            st.progress(confidence)

        st.divider()

        # --- Media Display ---
        st.subheader("Associated Media")
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("#### Query Media")
            q_img_path = metadata.get('query_image_path')
            q_cap_path = metadata.get('query_caption_path')
            if q_img_path and Path(q_img_path).exists():
                st.image(q_img_path, use_container_width=True)
            if q_cap_path and Path(q_cap_path).exists():
                st.caption(f"\"{Path(q_cap_path).read_text().strip()}\"")
        
        with col_img2:
            st.markdown("#### Top Visual Evidence")
            best_evidence_path = Path(metadata.get('evidences', [{}])[0].get('image_path', ''))
            if best_evidence_path.exists():
                st.image(str(best_evidence_path), use_container_width=True)
                evidence_caption = Path(metadata['evidences'][0]['caption_path']).read_text().strip()
                st.caption(f"\"{evidence_caption}\"")
            else:
                st.info("No visual evidence was found or used in this analysis.")
                
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch details for Query ID '{query_id}': {e}")