from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# --- Submodels for Parser ---
class SourceProvenance(BaseModel):
    source_title: str = Field(description="Title or identifier of the research source")
    source_url: str = Field(description="URL or database ID (e.g., PubMed ID) of the source")
    exact_snippet: str = Field(description="Verbatim text snippet containing the medical facts")
    retrieval_timestamp: str = Field(description="When the literature was fetched")

class SymptomExtraction(BaseModel):
    symptom_name: str = Field(description="Name of the symptom")
    description: str = Field(description="Raw text description of the symptom")
    background_text: Optional[str] = Field(description="Raw text mentioning background occurrence in healthy controls, if present")

class RawExtractedConcept(BaseModel):
    disease_name: str = Field(description="Name of the disease extracted from literature")
    prevalence_text: str = Field(description="Raw text snippet mentioning epidemiological prevalence")
    symptoms_observed: List[SymptomExtraction] = Field(default_factory=list, description="List of symptoms observed with background rates")
    provenance: SourceProvenance = Field(description="Source provenance metadata")

# --- Submodels for Translator ---
class SymptomProbability(BaseModel):
    id: str = Field(description="Normalized snake_case identifier for symptom")
    leak_probability: float = Field(ge=0.0, le=0.2, description="Background noise probability p0")
    description: str

class CausalEdgeDefinition(BaseModel):
    disease_id: str
    symptom_id: str
    p1: float = Field(ge=0.0, le=1.0, description="Causal mechanism strength p1")

class ProbabilisticConversionSchema(BaseModel):
    disease_id: str = Field(description="Normalized snake_case identifier for disease")
    prior: float = Field(ge=0.00001, le=0.95, description="Converted baseline prevalence P(D=1)")
    symptoms: List[SymptomProbability] = Field(default_factory=list, description="Symptoms with leak probabilities")
    edges: List[CausalEdgeDefinition] = Field(default_factory=list, description="Causal edges with mechanism strength")


class ValidationResultSchema(BaseModel):
    is_valid: bool = Field(description="Whether all probabilistic and structural rules passed")
    errors: List[str] = Field(default_factory=list, description="Specific feedback or mathematical violations if invalid")


# --- LangGraph Shared Graph State ---
class ExtractionGraphState(TypedDict):
    raw_literature_text: Optional[str]
    target_disease: Optional[str]
    target_symptom: Optional[str]
    batch_name: Optional[str]
    extracted_data: Optional[Dict[str, Any]]
    converted_payload: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    iteration_count: int
