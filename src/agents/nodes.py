import os
from pathlib import Path
import json
from dotenv import load_dotenv
from typing import Dict, Any
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from src.agents.state import (
    ExtractionGraphState, 
    RawExtractedConcept, 
    ProbabilisticConversionSchema, 
    ValidationResultSchema
)
from src.agents.prompts import (
    PARSER_SYSTEM_PROMPT, 
    TRANSLATOR_SYSTEM_PROMPT, 
    CRITIC_SYSTEM_PROMPT
)

load_dotenv()


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1)
    

def parser_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Agent 1: Formulates a query, searches live literature via Tavily, and extracts facts with provenance."""
    llm = get_llm()
    
    # Deterministic Python Query Formulation from keywords/input
    disease = state.get("target_disease", "COVID-19")
    symptom = state.get("target_symptom", "Loss of Taste ageusia")
    query = f"{disease} {symptom} clinical prevalence background rate medical study"
    print(f"\n🔍 Formulated Search Query: '{query}'")
    
    # Search Execution via Tavily API
    tavily_tool = TavilySearchResults(max_results=2)
    search_results = tavily_tool.invoke(query)
    
    # Aggregate text and capture source provenance metadata from search results
    literature_snippets = []
    source_title = "Unknown Source"
    source_url = "https://tavily.com"
    
    if search_results:
        # Take the top result as primary source provenance
        top_result = search_results[0]
        source_title = top_result.get("title", "Web Medical Literature")
        source_url = top_result.get("url", "")
        
        for idx, res in enumerate(search_results):
            literature_snippets.append(f"Source [{idx+1}] ({res.get('url')}):\n{res.get('content')}")
            
    combined_literature_text = "\n\n".join(literature_snippets)
    
    # Fallback if search returns nothing
    if not combined_literature_text:
        combined_literature_text = state.get("raw_literature_text", "No literature found.")

    # 3. LLM Extraction with Structured Output (incorporating provenance)
    structured_llm = llm.with_structured_output(RawExtractedConcept)
    
    prompt = f"""
    {PARSER_SYSTEM_PROMPT}
    
    Retrieved Literature Text:
    {combined_literature_text}
    """
    
    result = structured_llm.invoke(prompt)
    extracted_dict = result.dict()
    
    # Manually inject the captured source provenance metadata into the extracted dictionary
    extracted_dict["provenance"] = {
        "source_title": source_title,
        "source_url": source_url,
        "exact_snippet": combined_literature_text[:500], # Store preview snippet
        "retrieval_timestamp": datetime.now().isoformat()
    }

    state["extracted_data"] = extracted_dict
    return state



def translator_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Agent 2: Reads the target disease record from the aggregated domain raw file on disk."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(ProbabilisticConversionSchema)
    
    # 1. Determine batch name and target disease from state
    batch_name = state.get("batch_name", "clinical_extractions")
    target_disease = state.get("target_disease", "COVID-19")
    raw_file_path = Path("data") / f"{batch_name}_raw.json"
    
    extracted_data = {}
    
    # 2. Read directly from the domain raw file on disk and filter for the specific disease
    if raw_file_path.exists():
        try:
            with open(raw_file_path, "r", encoding="utf-8") as f:
                domain_data = json.load(f)
                extractions = domain_data.get("extractions", [])
                
                # Find the matching disease extraction in the batch list
                matching_extraction = next(
                    (item for item in extractions if item.get("disease_name", "").lower() == target_disease.lower()),
                    None
                )
                
                if matching_extraction:
                    extracted_data = matching_extraction
                    print(f"\n📂 Translator loaded extraction for '{target_disease}' from domain file: {raw_file_path}")
                else:
                    print(f"\n⚠️ Warning: Disease '{target_disease}' not found in {raw_file_path}. Falling back to state memory.")
                    extracted_data = state.get("extracted_data", {})
        except Exception as e:
            print(f"\n⚠️ Error reading domain raw file: {e}. Falling back to state memory.")
            extracted_data = state.get("extracted_data", {})
    else:
        print(f"\n⚠️ Warning: Domain raw file not found at {raw_file_path}. Falling back to state memory.")
        extracted_data = state.get("extracted_data", {})

    # 3. Handle critic feedback loop if iterating
    feedback = ""
    if state.get("validation_result") and not state["validation_result"].get("is_valid"):
        errors = "\n".join(state["validation_result"].get("errors", []))
        feedback = f"\n\nCRITIC FEEDBACK FROM PREVIOUS ATTEMPT (Fix these errors):\n{errors}"

    prompt = f"{TRANSLATOR_SYSTEM_PROMPT}{feedback}\n\nExtracted Data:\n{extracted_data}"

    result = structured_llm.invoke(prompt)
    state["converted_payload"] = result.dict()
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return state


def critic_node(state: ExtractionGraphState) -> ExtractionGraphState:
    """Validator: Programmatically checks math bounds. If math passes, approve immediately."""
    payload = state.get("converted_payload", {})
    errors = []
    
    # 1. Deterministic Python Math Checks
    prior = payload.get("prior", 0)
    if not (0.0 < prior < 1.0):
        errors.append(f"Prior probability ({prior}) must be between 0.0 and 1.0.")
        
    for symptom in payload.get("symptoms", []):
        p0 = symptom.get("leak_probability", 0)
        if not (0.0 <= p0 <= 1.0):
            errors.append(f"Leak probability ({p0}) must be between 0.0 and 1.0.")
            
    for edge in payload.get("edges", []):
        p1 = edge.get("p1", 0)
        if not (0.0 <= p1 <= 1.0):
            errors.append(f"Causal strength p1 ({p1}) must be between 0.0 and 1.0.")
            
        symptom_id = edge.get("symptom_id")
        matching_symptom = next((s for s in payload.get("symptoms", []) if s["id"] == symptom_id), None)
        if matching_symptom:
            p0 = matching_symptom.get("leak_probability", 0)
            if p1 <= p0:
                errors.append(f"Causal strength (p1 = {p1}) must be greater than background leak (p0 = {p0}).")

    # If programmatic math checks failed, mark invalid
    if errors:
        state["validation_result"] = {"is_valid": False, "errors": errors}
        return state

    # 2. If Python math passes, bypass LLM hallucination risk and approve!
    print("\n✅ Deterministic math checks passed successfully!")
    state["validation_result"] = {"is_valid": True, "errors": []}
    return state

