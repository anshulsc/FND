# pages/5_Extract_New_Evidence.py
import streamlit as st
from pathlib import Path
import time

from src.modules.evidence_agent import EvidenceExtractionAgent

st.set_page_config(page_title="Extract Evidence", page_icon="🕵️", layout="wide")
st.title("🕵️ Agentic Evidence Extractor")
st.info("Use this tool to automatically search the web for new evidence related to a topic and add it to your local database.")

# --- Agent Invocation Form ---
with st.form("evidence_agent_form"):
    st.header("Provide a Topic for the Agent to Investigate")
    
    image_file = st.file_uploader(
        "Upload a relevant Image (for reverse image search)",
        type=['png', 'jpg', 'jpeg', 'webp']
    )
    
    query_text = st.text_area(
        "Enter a News Caption or Topic (for text search)",
        height=100,
        placeholder="e.g., 'politician speaks at recent rally' or a full news headline"
    )
    
    submitted = st.form_submit_button("Start Evidence Extraction Agent", use_container_width=True, type="primary")

if submitted:
    if not image_file or not query_text.strip():
        st.error("Please provide both an image and a text query to run the full pipeline.")
    else:
        # Save the uploaded image temporarily to pass its path to the agent
        temp_dir = Path("./temp_agent_run")
        temp_dir.mkdir(exist_ok=True)
        temp_image_path = temp_dir / image_file.name
        
        with open(temp_image_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        st.warning("Agent is running... This may take a minute. Live output will be shown below.", icon="🤖")
        
        log_container = st.empty()
        log_content = ""

        try:
            agent = EvidenceExtractionAgent()
            
            # --- Text Search ---
            log_content += f"[AGENT] Starting text search for: '{query_text}'\n"
            log_container.code(log_content, language="log")
            added_text = agent.run_text_search(query_text)
            log_content += f"[AGENT] Text search complete. Added {added_text} new items.\n\n"
            log_container.code(log_content, language="log")
            
            time.sleep(1) # Small pause for better UX

            # --- Image Search ---
            log_content += f"[AGENT] Starting reverse image search for: {image_file.name}\n"
            log_container.code(log_content, language="log")
            added_img = agent.run_reverse_image_search(str(temp_image_path))
            log_content += f"[AGENT] Reverse image search complete. Added {added_img} new items.\n"
            log_container.code(log_content, language="log")
            
            total = added_text + added_img
            st.success(f"🎉 Agent pipeline finished! A total of {total} new evidence items were added to the database.", icon="✅")
            st.info("You can now run the 'Re-Index Evidence Database' command on the Settings page to make this new evidence searchable.")

        except Exception as e:
            st.error(f"An error occurred while running the agent: {e}")
        finally:
            # Clean up the temporary directory
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)