# src/modules/evidence_searcher.py
import chromadb
from pathlib import Path

from src.config import VECTOR_DB_DIR, QUERIES_DIR, EVIDENCE_DB_DIR
from src.modules.embedding_utils import get_image_embedding, get_text_embedding

# --- ChromaDB Client Setup ---
client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
collection = client.get_collection(name="evidence_collection")

def find_top_evidence(query_image_path: str, query_caption: str, top_k: int = 10):
    """
    Finds the most similar evidence from the vector database for a given query.

    Args:
        query_image_path (str): Path to the query image.
        query_caption (str): The query caption text.
        top_k (int): The number of top results to return.

    Returns:
        A list of dictionaries, where each dictionary represents a piece of
        evidence with its score and paths.
    """
    print(f"INFO: Searching for evidence for query image: {query_image_path}")

    # 1. Generate embeddings for the query
    query_img_emb = get_image_embedding(query_image_path)
    query_text_emb = get_text_embedding(query_caption)

    if query_img_emb is None or query_text_emb is None:
        print("ERROR: Failed to generate embeddings for the query. Aborting search.")
        return []

    # 2. Perform two separate searches: one for images, one for text
    image_results = collection.query(
        query_embeddings=[query_img_emb],
        n_results=top_k,
        where={"type": "image"} # Filter for image embeddings
    )

    text_results = collection.query(
        query_embeddings=[query_text_emb],
        n_results=top_k,
        where={"type": "text"} # Filter for text embeddings
    )

    # 3. Combine and re-rank the results
    # We merge results to get the best matches from both modalities
    combined_results = {}

    # Distances from ChromaDB are squared L2 distances. Lower is better.
    # We'll convert them to a similarity score (0 to 1) for easier interpretation.
    # similarity = 1 - distance
    
    # Process image results
    for i, item_id in enumerate(image_results['ids'][0]):
        base_id = item_id.replace('_img', '')
        distance = image_results['distances'][0][i]
        similarity = max(0, 1 - distance) # Ensure similarity is not negative
        
        if base_id not in combined_results or similarity > combined_results[base_id]['similarity_score']:
             combined_results[base_id] = {
                'similarity_score': similarity,
                'path': image_results['metadatas'][0][i]['path']
            }

    # Process text results
    for i, item_id in enumerate(text_results['ids'][0]):
        base_id = item_id.replace('_txt', '')
        distance = text_results['distances'][0][i]
        similarity = max(0, 1 - distance)
        
        if base_id not in combined_results or similarity > combined_results[base_id]['similarity_score']:
            combined_results[base_id] = {
                'similarity_score': similarity,
                'path': text_results['metadatas'][0][i]['path']
            }
            
    # 4. Sort the combined results by similarity score (highest first)
    sorted_evidence = sorted(combined_results.items(), key=lambda item: item[1]['similarity_score'], reverse=True)

    # 5. Format the final output
    final_results = []
    for rank, (item_id, data) in enumerate(sorted_evidence[:top_k], 1):
        # We need to find the corresponding image and caption paths for the item_id
        evidence_dir = EVIDENCE_DB_DIR / item_id
        img_path = next(evidence_dir.glob('*.jpg'), None) # Add other extensions if needed
        cap_path = next(evidence_dir.glob('*.txt'), None)
        
        if img_path and cap_path:
            final_results.append({
                "rank": rank,
                "similarity_score": round(data['similarity_score'], 4),
                "image_path": str(img_path.resolve()),
                "caption_path": str(cap_path.resolve())
            })

    print(f"INFO: Found {len(final_results)} relevant evidence items.")
    return final_results

if __name__ == "__main__":
    # --- This block allows us to test the searcher directly ---
    print("--- Running Standalone Evidence Searcher Test ---")
    
    # Use an existing query to test. Make sure this folder exists.
    test_query_dir = QUERIES_DIR / "0"
    
    # Find the image and caption files
    try:
        test_image_path = next(test_query_dir.glob('*.jpg')) # Add other extensions if needed
        test_caption_path = next(test_query_dir.glob('*.txt'))
        
        with open(test_caption_path, 'r') as f:
            test_caption = f.read().strip()
            
        print(f"Test Query Image: {test_image_path}")
        print(f"Test Query Caption: '{test_caption}'")
        
        # Run the search
        top_evidence = find_top_evidence(str(test_image_path), test_caption, top_k=5)
        
        # Print results
        if top_evidence:
            print("\n--- Top 5 Evidence Found ---")
            for item in top_evidence:
                print(
                    f"Rank: {item['rank']}, "
                    f"Score: {item['similarity_score']}, "
                    f"Image: ...{Path(item['image_path']).name}, "
                    f"Caption: ...{Path(item['caption_path']).parent.name}/{Path(item['caption_path']).name}"
                )
        else:
            print("No evidence found.")
            
    except (StopIteration, FileNotFoundError):
        print("\nERROR: Test query not found!")
        print(f"Please make sure the directory '{test_query_dir}' exists and contains an image and a .txt file.")