# src/modules/inference_pipeline.py
import json
import torch
from pathlib import Path

# Import your adapted, original logic
from src.config import VLLM_MODEL_PATH, FRAUDNET_MODEL_PATH, DOMAIN_VECTOR_PATH
from src.agents.agent_class import MultimodalClaimVerifier
from src.fraudnet import load_model, extract_clip_features, load_domain_vector, fraudnet_input
from src.workflow import build_langgraph

# --- Initialize Models (Singleton Pattern) ---
# This ensures these heavy models are loaded only ONCE when the worker starts.
_verifier = None
_fraudnet_model = None
_domain_vec = None
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def _initialize_models():
    """Loads and caches all the necessary AI models."""
    global _verifier, _fraudnet_model, _domain_vec
    
    if _verifier is None:
        print("INFO: Initializing MultimodalClaimVerifier (vLLM)...")
        _verifier = MultimodalClaimVerifier(VLLM_MODEL_PATH)
        print("INFO: Verifier initialized.")

    if _fraudnet_model is None:
        print("INFO: Initializing FraudNet model...")
        _fraudnet_model = load_model(FRAUDNET_MODEL_PATH, device=DEVICE)
        _domain_vec = load_domain_vector(DOMAIN_VECTOR_PATH, device=DEVICE)
        print("INFO: FraudNet model initialized.")

def run_full_inference(metadata_path: Path):
    """
    Takes the path to an evidence_metadata.json file, runs the full
    LangGraph and FraudNet pipeline, and saves the results.
    """
    # 1. Ensure models are loaded
    _initialize_models()

    # 2. Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    query_image_path = metadata['query_image_path']
    evidence_image_path = metadata['evidences'][0]['image_path'] if metadata['evidences'] else query_image_path # Fallback
    
    with open(metadata['query_caption_path'], 'r') as f:
        query_caption = f.read().strip()

    # 3. Prepare inputs in the format your original code expects
    # The original code expected newline-separated strings. We adapt our structured
    # evidence to fit that legacy format.
    search_results = "\n".join([Path(e['caption_path']).read_text().strip() for e in metadata['evidences']])
    # The original logic used the query caption as the claim for each piece of evidence.
    claims = "\n".join([query_caption for _ in metadata['evidences']])
    
    # This logic is adapted from your original get_txt_txt_input function
    txt_txt_inputs = [
        (sr, cl) for sr, cl in zip(search_results.split("\n"), claims.split("\n")) if sr and cl
    ]

    # 4. Prepare FraudNet inputs (adapted from your original app.py)
    img_feat, text_feat, X_all = extract_clip_features(query_image_path, query_caption, evidence_image_path, search_results)
    img_feat, text_feat = img_feat.to(DEVICE), text_feat.to(DEVICE)
    fraudnet_inputs = fraudnet_input(
        img_feat=img_feat.unsqueeze(0),
        text_feat=text_feat.unsqueeze(0),
        domain_vec=_domain_vec.unsqueeze(0),
        fake_evidence=X_all.unsqueeze(0).to(DEVICE)
    )

    # 5. Build the LangGraph state and invoke the graph
    # This part is directly from your original get_results function
    state = {
        'query_image_path': query_image_path,
        'evidence_image_path': evidence_image_path,
        'query_caption': query_caption,
        'txt_txt_inputs': txt_txt_inputs,
        'verifier': _verifier,
        'fraudnet_model': _fraudnet_model,
        'fraudnet_input': fraudnet_inputs
    }
    
    graph = build_langgraph()
    final_state = graph.invoke(state)

    # Clean up the result to be JSON serializable (remove non-serializable objects)
    final_output = {
        key: final_state[key] for key in ["stage2_outputs", "fraudnet_response"] if key in final_state
    }

    # 6. Save the results to a new JSON file
    result_path = metadata_path.parent / "inference_results.json"
    with open(result_path, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"INFO: Inference complete. Results saved to {result_path}")
    return result_path