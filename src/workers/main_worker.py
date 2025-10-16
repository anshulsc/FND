# src/workers/main_worker.py (Updated)
import os
import time
import shutil
import json
from pathlib import Path

from src.config import (
    JOB_QUEUE_DIR, JOB_COMPLETED_DIR, JOB_FAILED_DIR, WORKER_SLEEP_INTERVAL,
    QUERIES_DIR, PROCESSED_DIR
)
from src.database.status_manager import status_manager
# Import our new searcher
from src.modules.evidence_searcher import find_top_evidence
from src.modules.inference_pipeline import run_full_inference
from src.modules.pdf_generator import create_report_pdf
from src.logger_config import worker_logger

def find_query_files(query_id: str):
    """Finds the image and caption file for a given query_id."""
    query_path = QUERIES_DIR / query_id
    if not query_path.is_dir():
        raise FileNotFoundError(f"Query directory not found: {query_path}")
    
    try:
        img_file = next(query_path.glob('*.[jp][pn]g')) # jpg, jpeg, png
    except StopIteration:
        img_file = next(query_path.glob('*.webp')) # Add other types if needed
    
    cap_file = next(query_path.glob('*.txt'))
    
    return img_file, cap_file

def process_job(job_path):
    """
    This is the core logic for processing a single job.
    """
    query_id = job_path.stem
    worker_logger.info(f"\n--- [WORKER] Processing job for query: {query_id} ---")

    try:
        # STAGE 1: Evidence Extraction (REAL IMPLEMENTATION)
        status_manager.update_stage_status(query_id, "evidence_extraction", "processing")
        worker_logger.info(f"INFO: [Stage 1/3] Starting Evidence Extraction for '{query_id}'...")
        
        # 1. Find query files
        q_img_path, q_cap_path = find_query_files(query_id)
        with open(q_cap_path) as f:
            q_caption = f.read().strip()

        # 2. Run the search
        evidence_results = find_top_evidence(str(q_img_path), q_caption)

        # 3. Prepare the processed directory
        processed_query_dir = PROCESSED_DIR / query_id
        processed_query_dir.mkdir(exist_ok=True)
        
        # 4. Copy original query files
        shutil.copy(q_img_path, processed_query_dir / q_img_path.name)
        shutil.copy(q_cap_path, processed_query_dir / q_cap_path.name)
        
        # 5. Copy best evidence image
        if evidence_results:
            best_evidence_path = Path(evidence_results[0]['image_path'])
            shutil.copy(best_evidence_path, processed_query_dir / "best_evidence.jpg")
        
        # 6. Save the metadata file
        metadata = {
            "query_id": query_id,
            "query_image_path": str((processed_query_dir / q_img_path.name).resolve()),
            "query_caption_path": str((processed_query_dir / q_cap_path.name).resolve()),
            "evidences": evidence_results
        }
        metadata_path = processed_query_dir / "evidence_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        worker_logger.info(f"INFO: Evidence extraction complete. Metadata saved to {metadata_path}")
        status_manager.update_stage_status(query_id, "evidence_extraction", "completed")
        
        # STAGE 2: Model Inference (Placeholder)
        status_manager.update_stage_status(query_id, "model_inference", "processing")
        current_stage = "model_inference"
        status_manager.update_stage_status(query_id, "model_inference", "processing")
        worker_logger.info(f"INFO: [Stage 2/3] Starting Model Inference for '{query_id}'...")
        
        inference_result_path = run_full_inference(metadata_path)
        status_manager.update_stage_status(query_id, "model_inference", "completed")

        # STAGE 3: PDF Generation (Placeholder)
        current_stage = "pdf_generation"
        status_manager.update_stage_status(query_id, "pdf_generation", "processing")
        worker_logger.info(f"INFO: [Stage 3/3] Starting PDF Generation for '{query_id}'...")
        
        pdf_path = create_report_pdf(metadata_path, inference_result_path)
        status_manager.set_result_path(query_id, str(pdf_path.resolve()))

        status_manager.update_stage_status(query_id, "pdf_generation", "completed")

        worker_logger.info(f"SUCCESS: Job for '{query_id}' completed successfully.")
        return True


    except Exception as e:
        import traceback
        error_msg = f"ERROR: Job for '{query_id}' failed at stage '{current_stage}'. Reason: {e}"
        worker_logger.info(error_msg)
        traceback.worker_logger.info_exc() # worker_logger.info full error for debugging
        status_manager.update_stage_status(query_id, current_stage, "failed", error_message=str(e))
        return False

# The rest of main_worker.py (start_worker function) remains the same
def start_worker():
    """Starts the main worker loop to check for and process jobs."""
    worker_logger.info("--- Starting Main Worker ---")
    worker_logger.info(f"Watching for jobs in: {JOB_QUEUE_DIR}")

    while True:
        job_files = list(JOB_QUEUE_DIR.glob("*.job"))
        if not job_files:
            time.sleep(WORKER_SLEEP_INTERVAL)
            continue

        job_path = job_files[0]
        is_successful = process_job(job_path)

        if is_successful:
            destination = JOB_COMPLETED_DIR / job_path.name
        else:
            destination = JOB_FAILED_DIR / job_path.name
            
        shutil.move(str(job_path), str(destination))
        worker_logger.info(f"INFO: Moved job file to {destination}")

if __name__ == "__main__":
    try:
        start_worker()
    except KeyboardInterrupt:
        worker_logger.info("\n--- Worker stopped ---")