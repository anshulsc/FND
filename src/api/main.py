# src/api/main.py
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import List
import json
import regex as re


from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware


from src.config import QUERIES_DIR, JOB_QUEUE_DIR, RESULTS_DIR, PROCESSED_DIR, TRASH_DIR
from src.database.status_manager import status_manager
from src.logger_config import api_logger

app = FastAPI(title="Agentic Framework API")

# Allow CORS for Streamlit communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Streamlit app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _extract_verdict_from_results(query_id: str) -> str:
    """A helper function to safely read the results file and extract the verdict."""
    try:
        results_path = PROCESSED_DIR / query_id / "inference_results.json"
        if not results_path.exists():
            return "N/A"

        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        final_response = results.get('stage2_outputs', {}).get('final_response', "")
        
        verdict_match = re.search(r"\*\*Final Classification\*\*:\s*(\w+)", final_response, re.IGNORECASE)
        if verdict_match:
            verdict = verdict_match.group(1).upper()
            if "FAKE" in verdict:
                return "Fake"
            elif "TRUE" in verdict or "REAL" in verdict:
                return "True"
        return "Uncertain"
    except Exception:
        return "Error"

@app.get("/queries", summary="Get status of all queries")
def get_all_queries():
    """Returns a list of all queries with their status and final verdict if available."""
    api_logger.info("Request received for /queries endpoint.")
    queries = status_manager.get_all_queries()
    
    enriched_queries = []
    for query in queries:
        query_dict = dict(query) 
        if query_dict['status'] == 'completed':
            query_dict['verdict'] = _extract_verdict_from_results(query_dict['query_id'])
        else:
            query_dict['verdict'] = "Pending"
        enriched_queries.append(query_dict)
            
    return JSONResponse(content={"queries": enriched_queries})

@app.get("/results/{query_id}", summary="Get a PDF report")
def get_result_pdf(query_id: str):
    """Serves the final PDF report for a given query ID."""
    query_info = status_manager.get_query_status(query_id)
    if not query_info or not query_info.get("result_pdf_path"):
        api_logger.warning("Result PDF not found or not yet generated", extra={"query_id": query_id})
        raise HTTPException(status_code=404, detail="Result PDF not found or not yet generated.")
    
    pdf_path = Path(query_info["result_pdf_path"])
    if not pdf_path.exists():
        api_logger.error("PDF file missing from filesystem", extra={"query_id": query_id, "path": str(pdf_path)})
        raise HTTPException(status_code=404, detail="PDF file is missing from the filesystem.")

    api_logger.info("Serving result PDF", extra={"query_id": query_id, "path": str(pdf_path)})
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{query_id}_report.pdf")

@app.post("/rerun/{query_id}", summary="Rerun a query")
def rerun_query(query_id: str):
    """Resets a query and adds it back to the job queue for reprocessing."""
    if not (QUERIES_DIR / query_id).exists():
        api_logger.warning("Query ID not found in queries directory", extra={"query_id": query_id})
        raise HTTPException(status_code=404, detail=f"Query ID '{query_id}' not found in queries directory.")

    # Reset status in DB
    status_manager.reset_query(query_id)
    
    # Create job file to trigger worker
    job_file_path = JOB_QUEUE_DIR / f"{query_id}.job"
    job_file_path.touch()
    
    api_logger.info("Query queued for rerun", extra={"query_id": query_id, "job_file": str(job_file_path)})
    return JSONResponse(content={"message": f"Query '{query_id}' has been queued for rerun."})

@app.post("/add_query_manual", summary="Add query via image and text")
async def add_query_manual(caption: str = Form(...), image: UploadFile = File(...)):
    """Creates a new query from an uploaded image and caption text."""
    query_id = f"query_{uuid.uuid4().hex[:8]}"
    query_dir = QUERIES_DIR / query_id
    query_dir.mkdir()

    # Save image
    image_ext = Path(image.filename).suffix or ".jpg"
    image_path = query_dir / f"query_img{image_ext}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Save caption
    caption_path = query_dir / "query_cap.txt"
    with open(caption_path, "w") as f:
        f.write(caption)

    # The watcher will automatically pick this up. The API's job is done.
    api_logger.info("Manual query added", extra={"query_id": query_id, "image_path": str(image_path), "caption_path": str(caption_path)})
    return JSONResponse(content={"message": "Query added successfully.", "query_id": query_id})

@app.post("/add_query_folder", summary="Add query via zipped folder")
async def add_query_folder(file: UploadFile = File(...)):
    """Creates a new query from an uploaded zip file containing query_img and query_cap."""
    if not file.filename.endswith('.zip'):
        api_logger.warning("Invalid file type for add_query_folder", extra={"filename": file.filename})
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .zip file.")

    query_id = f"query_{uuid.uuid4().hex[:8]}"
    query_dir = QUERIES_DIR / query_id
    temp_zip_path = query_dir.with_suffix('.zip')

    # Save and extract zip file
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(query_dir)
    
    # Clean up temp file
    temp_zip_path.unlink()
    
    # The watcher will automatically pick this up.
    api_logger.info("Folder query uploaded and extracted", extra={"query_id": query_id, "extract_dir": str(query_dir)})
    return JSONResponse(content={"message": "Query folder uploaded and extracted successfully.", "query_id": query_id})

@app.get("/details/{query_id}", summary="Get full JSON details for a query")
def get_query_details(query_id: str):
    """Returns the full inference_results.json for a given query ID."""
    results_path = PROCESSED_DIR / query_id / "inference_results.json"
    api_logger.debug("Fetching query details", extra={"query_id": query_id, "results_path": str(results_path)})
    
    if not results_path.exists():
        api_logger.warning("Inference results JSON not found", extra={"query_id": query_id, "results_path": str(results_path)})
        raise HTTPException(status_code=404, detail="Inference results JSON file not found.")
        
    query_info = status_manager.get_query_status(query_id)
    metadata_path = PROCESSED_DIR / query_id / "evidence_metadata.json"

    details = {
        "status": query_info,
        "results": json.loads(results_path.read_text()),
        "metadata": json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    }

    api_logger.info("Returning query details", extra={"query_id": query_id, "has_metadata": metadata_path.exists()})
    return JSONResponse(content=details)

@app.delete("/trash/{query_id}", summary="Move a query and its results to trash")
def move_query_to_trash(query_id: str):
    """Moves a query's processed files and results to the trash folder and updates its status."""
    api_logger.info(f"Received request to move query '{query_id}' to trash.")
    
    # Define source paths
    processed_path = PROCESSED_DIR / query_id
    results_path = RESULTS_DIR / query_id
    
    # Define destination paths
    trash_processed_path = TRASH_DIR / "processed" / query_id
    trash_results_path = TRASH_DIR / "results" / query_id
    
    try:
        if processed_path.exists():
            shutil.move(str(processed_path), str(trash_processed_path))
        if results_path.exists():
            shutil.move(str(results_path), str(trash_results_path))
            
        # Update the status in the database
        status_manager.move_to_trash(query_id)
        
        return JSONResponse(content={"message": f"Query '{query_id}' moved to trash."})
    except Exception as e:
        api_logger.error(f"Failed to move '{query_id}' to trash: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to move to trash: {e}")

@app.post("/restore/{query_id}", summary="Restore a query from trash")
def restore_query_from_trash(query_id: str):
    """Moves a query's files back from the trash and resets its status to pending."""
    api_logger.info(f"Received request to restore query '{query_id}' from trash.")

    # Define source paths in trash
    trash_processed_path = TRASH_DIR / "processed" / query_id
    trash_results_path = TRASH_DIR / "results" / query_id

    # Define original destination paths
    processed_path = PROCESSED_DIR / query_id
    results_path = RESULTS_DIR / query_id
    
    try:
        if trash_processed_path.exists():
            shutil.move(str(trash_processed_path), str(processed_path))
        if trash_results_path.exists():
            shutil.move(str(trash_results_path), str(results_path))
        
        # Reset the status in the database (restoring puts it back in a neutral state)
        status_manager.reset_query(query_id)
        # We create a job file so the user can choose to rerun it from the dashboard.
        (JOB_QUEUE_DIR / f"{query_id}.job").touch()
        
        return JSONResponse(content={"message": f"Query '{query_id}' restored and queued for processing."})
    except Exception as e:
        api_logger.error(f"Failed to restore '{query_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to restore from trash: {e}")

@app.delete("/delete_permanent/{query_id}", summary="Permanently delete a query")
def delete_query_permanently(query_id: str):
    """Permanently deletes a query's files from trash and its record from the database."""
    api_logger.warning(f"Received request to PERMANENTLY DELETE query '{query_id}'.")

    # Define paths in trash
    trash_processed_path = TRASH_DIR / "processed" / query_id
    trash_results_path = TRASH_DIR / "results" / query_id
    
    try:
        if trash_processed_path.exists():
            shutil.rmtree(trash_processed_path)
        if trash_results_path.exists():
            shutil.rmtree(trash_results_path)
            
        # Remove from the database
        status_manager.delete_permanently(query_id)
        
        return JSONResponse(content={"message": f"Query '{query_id}' has been permanently deleted."})
    except Exception as e:
        api_logger.error(f"Failed to permanently delete '{query_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to permanently delete: {e}")

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    api_logger.info("Starting API server", extra={"host": API_HOST, "port": API_PORT})
    uvicorn.run(app, host=API_HOST, port=API_PORT)