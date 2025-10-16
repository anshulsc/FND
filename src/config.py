# src/config.py
import os
from pathlib import Path

# --- Core Paths ---
# Use pathlib for robust path handling across different OS
BASE_DIR = Path(__file__).parent.parent.resolve()
print(f"Base Directory: {BASE_DIR}")
WORKSPACE_DIR = BASE_DIR / "agentic_workspace"

# --- Workspace Directories ---
QUERIES_DIR = WORKSPACE_DIR / "1_queries"
EVIDENCE_DB_DIR = WORKSPACE_DIR / "2_evidence_database"
PROCESSED_DIR = WORKSPACE_DIR / "3_processed_for_model"
RESULTS_DIR = WORKSPACE_DIR / "4_results"
TRASH_DIR = WORKSPACE_DIR / "5_trash"

# --- System Directories ---
SYSTEM_DIR = WORKSPACE_DIR / ".system"
LOGS_DIR = SYSTEM_DIR / "logs"
JOB_QUEUE_DIR = SYSTEM_DIR / "job_queue"
JOB_COMPLETED_DIR = SYSTEM_DIR / "job_completed"
JOB_FAILED_DIR = SYSTEM_DIR / "job_failed"
VECTOR_DB_DIR = SYSTEM_DIR / "vector_db"
SEARCH_INDEX_DIR = SYSTEM_DIR / "search_index"
DB_PATH = SYSTEM_DIR / "app_state.db"

# --- Create Directories if they don't exist ---
# This makes the setup robust
for path in [
    QUERIES_DIR, EVIDENCE_DB_DIR, PROCESSED_DIR, RESULTS_DIR, TRASH_DIR,
    SYSTEM_DIR, LOGS_DIR, JOB_QUEUE_DIR, JOB_COMPLETED_DIR, JOB_FAILED_DIR,
    VECTOR_DB_DIR, SEARCH_INDEX_DIR
]:
    path.mkdir(parents=True, exist_ok=True)


# --- Model and API Configuration ---
# You can centralize other settings here as well
VLLM_MODEL_PATH = "google/gemma-3-12b-it" # Or your local path
FRAUDNET_MODEL_PATH = "/data/shwetabh_agentic_FND/agentic_framework_1/Indian_finetuned_AL2_180_may21_iteration_4.pth.tar"
DOMAIN_VECTOR_PATH = "/data/shwetabh_agentic_FND/agentic_framework_1/domain_vector_VITL14.json"

API_HOST = "0.0.0.0"
API_PORT = 8000

# --- Worker Configuration ---
# How often the worker checks for new jobs, in seconds
WORKER_SLEEP_INTERVAL = 5

BRAVE_API_KEY = "BSA1cZFR9cSOtpkalB5rDvwNEbOvZz9"
AGENT_SEARCH_RESULT_COUNT = 15