import os
import json
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from src.agents.state import ExtractionGraphState
from src.agents.nodes import parser_node, translator_node, critic_node

def save_raw_extraction_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Saves raw extractions into a domain-level aggregated JSON file."""
    os.makedirs("data", exist_ok=True)
    batch_name = state.get("batch_name", "clinical_extractions")
    file_path = os.path.join("data", f"{batch_name}_raw.json")
    
    # Load existing domain extractions or initialize structure
    domain_data = {"batch_name": batch_name, "extractions": []}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                domain_data = json.load(f)
        except json.JSONDecodeError:
            pass
            
    current_extraction = state.get("extracted_data", {})
    disease_name = current_extraction.get("disease_name", "unknown")
    
    # Upsert: Replace if disease already exists, otherwise append
    extractions = domain_data.get("extractions", [])
    extractions = [e for e in extractions if e.get("disease_name") != disease_name]
    extractions.append(current_extraction)
    domain_data["extractions"] = extractions
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(domain_data, f, indent=4)
        
    print(f"\n💾 Aggregated raw extraction saved to domain file: {file_path}")
    return state

def save_kb_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Saves validated knowledge base payloads into a domain-level aggregated JSON file."""
    os.makedirs("data", exist_ok=True)
    batch_name = state.get("batch_name", "clinical_extractions")
    file_path = os.path.join("data", f"{batch_name}_kb.json")
    
    domain_kb = {"domain": batch_name, "diseases": []}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                domain_kb = json.load(f)
        except json.JSONDecodeError:
            pass
            
    current_payload = state.get("converted_payload", {})
    disease_id = current_payload.get("disease_id", "unknown")
    
    # Upsert: Replace if disease already exists, otherwise append
    diseases = domain_kb.get("diseases", [])
    diseases = [d for d in diseases if d.get("disease_id") != disease_id]
    diseases.append(current_payload)
    domain_kb["diseases"] = diseases
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(domain_kb, f, indent=4)
        
    print(f"\n✅ Aggregated Knowledge Base saved to domain file: {file_path}")
    return state

def should_continue(state: ExtractionGraphState) -> str:
    """Conditional router that checks critic validation results and iteration limits."""
    validation = state.get("validation_result", {})
    print(f"\n🔍 Critic Feedback History: {validation.get('errors', [])}")
    print(f"🔍 Critic is_valid status: {validation.get('is_valid')}")
    
    is_valid = validation.get("is_valid", False)
    iteration_count = state.get("iteration_count", 0)
    
    if is_valid:
        print("\n✅ Validation passed successfully! Routing to save.")
        return "save"
    elif iteration_count >= 3:
        print("\n⚠️ Maximum critique iterations reached. Stopping pipeline.")
        return "stop"
    else:
        print(f"\n🔄 Validation failed. Routing back to translator for iteration {iteration_count + 1}...")
        return "retry"

def build_extraction_pipeline():
    """Builds and compiles the LangGraph state machine orchestrator."""
    workflow = StateGraph(ExtractionGraphState)
    
    # Add Nodes (including dedicated storage nodes)
    workflow.add_node("parser", parser_node)
    workflow.add_node("save_raw_extraction", save_raw_extraction_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("save_kb", save_kb_node)
    
    # Add Linear Edges
    workflow.add_edge(START, "parser")
    workflow.add_edge("parser", "save_raw_extraction")
    workflow.add_edge("save_raw_extraction", "translator")
    workflow.add_edge("translator", "critic")
    
    # Add Conditional Routing (Critique Loop)
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "save": "save_kb",
            "retry": "translator",
            "stop": END
        }
    )
    
    workflow.add_edge("save_kb", END)
    
    return workflow.compile()


def run_batch_pipeline(config_path: str = "data/queries.json"):
    """Loads query targets from external JSON config and runs domain batch extraction."""
    pipeline = build_extraction_pipeline()
    path = Path(config_path)
    
    if path.exists():
        with open(path, "r") as f:
            config = json.load(f)
            
        batch_name = config.get("batch_name", "clinical_extractions")
        os.makedirs("data", exist_ok=True)
        
        # Fresh Start: Reset/initialize domain files at the beginning of the batch run
        raw_file_path = os.path.join("data", f"{batch_name}_raw.json")
        kb_file_path = os.path.join("data", f"{batch_name}_kb.json")
        
        with open(raw_file_path, "w", encoding="utf-8") as f:
            json.dump({"batch_name": batch_name, "extractions": []}, f, indent=4)
            
        with open(kb_file_path, "w", encoding="utf-8") as f:
            json.dump({"domain": batch_name, "diseases": []}, f, indent=4)
            
        print(f"🧹 Initialized fresh domain files for batch: '{batch_name}'")
        print(f"🚀 Running batch extraction ({len(config.get('targets', []))} targets)...")
        
        # Loop through targets and append/upsert into the clean domain files
        for target in config.get("targets", []):
            disease = target["target_disease"]
            symptom = target["target_symptom"]
            
            print(f"\n{'='*50}\n🎯 Processing Target: {disease} -> {symptom}\n{'='*50}")
            
            initial_state = {
                "raw_literature_text": None,
                "target_disease": disease,
                "target_symptom": symptom,
                "batch_name": batch_name,
                "extracted_data": None,
                "converted_payload": None,
                "validation_result": None,
                "iteration_count": 0
            }
            pipeline.invoke(initial_state)
    else:
        print("⚠️ queries.json not found.")
        

# --- PIPELINE RUNNER ---
if __name__ == "__main__":
    print("🚀 Initializing Agentic Knowledge Extraction Pipeline...")
    run_batch_pipeline()
    print("\n🏁 Pipeline execution completed successfully!")
