# src/workers/watcher.py
import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.config import QUERIES_DIR, JOB_QUEUE_DIR
from src.database.status_manager import status_manager
from src.logger_config import watcher_logger

class QueryHandler(FileSystemEventHandler):
    """A handler for file system events in the queries directory."""

    def on_created(self, event):
        # We only care about new directories being created.
        if event.is_directory:
            query_id = os.path.basename(event.src_path)
            
            watcher_logger.info(f"INFO: Detected new query directory: {query_id}")
            
            # 1. Add the new query to our status database
            status_manager.add_query(query_id)
            
            # 2. Create a job file to trigger the worker
            job_file_path = JOB_QUEUE_DIR / f"{query_id}.job"
            with open(job_file_path, 'w') as f:
                f.write(query_id) # You can write content if needed later
            
            watcher_logger.info(f"INFO: Created job file for '{query_id}' at {job_file_path}")

def start_watcher():
    """Starts the file system watcher."""
    watcher_logger.info("--- Starting Query Watcher ---")
    watcher_logger.info(f"Monitoring directory: {QUERIES_DIR}")
    
    event_handler = QueryHandler()
    observer = Observer()
    observer.schedule(event_handler, str(QUERIES_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        watcher_logger.info("\n--- Watcher stopped ---")
    observer.join()

if __name__ == "__main__":
    start_watcher()