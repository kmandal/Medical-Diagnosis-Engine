import json
from pathlib import Path
import networkx as nx

class CausalGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_disease_node(self, disease_id: str, prior_prob: float, description: str = ""):
        """Adds a root disease node with its epidemiological prior P(D=1)."""
        self.graph.add_node(
            disease_id,
            node_type="disease",
            prior=prior_prob,
            description=description
        )

    def add_symptom_node(self, symptom_id: str, leak_probability: float = 0.02, description: str = ""):
        """Adds a leaf symptom node with background leak (p0)."""
        self.graph.add_node(
            symptom_id,
            node_type="symptom",
            leak_probability=leak_probability,
            description=description
        )
        
        # Connect 'Unknown_Condition' to every symptom with flat causal strength p1=0.30
        if self.graph.has_node("Unknown_Condition") and symptom_id != "Unknown_Condition":
            self.graph.add_edge("Unknown_Condition", symptom_id, p1=0.30)

    def add_causal_edge(self, disease_id: str, symptom_id: str, p_symptom_given_disease: float):
        """Adds a directed causal link: Disease -> Symptom with causal strength p1."""
        if not self.graph.has_node(disease_id) or not self.graph.has_node(symptom_id):
            raise ValueError(f"Both disease ({disease_id}) and symptom ({symptom_id}) nodes must exist in graph first.")
        self.graph.add_edge(disease_id, symptom_id, p1=p_symptom_given_disease)

    def setup_unknown_disease(self, unknown_prior: float = 0.001):
        """Initializes Unknown_Condition for novel/unmodeled pathologies."""
        self.add_disease_node("Unknown_Condition", prior_prob=unknown_prior, description="Unmodeled disease")

    def build_from_knowledge_base(self, kb_data: dict) -> nx.DiGraph:
        """
        Parses domain knowledge base JSON and builds the NetworkX graph,
        returning a fully constructed nx.DiGraph object.
        """
        # Setup Unknown Condition
        self.setup_unknown_disease(unknown_prior=0.001)
        
        added_symptoms = set()
        diseases = kb_data.get("diseases", [])

        # Iterate through diseases in KB
        for disease in diseases:
            d_id = disease["disease_id"]
            prior = disease["prior"]
            self.add_disease_node(d_id, prior_prob=prior)
            
            # Add symptoms with background leaks
            for symptom in disease.get("symptoms", []):
                s_id = symptom["id"]
                p0 = symptom.get("leak_probability", 0.02)
                if s_id not in added_symptoms:
                    self.add_symptom_node(s_id, leak_probability=p0)
                    added_symptoms.add(s_id)
                    
            # Add causal edges
            for edge in disease.get("edges", []):
                self.add_causal_edge(edge["disease_id"], edge["symptom_id"], edge["p1"])
                
        print(f"✅ Successfully built Causal Graph for domain: '{kb_data.get('domain', 'unknown')}'")
        
        return self.graph


    def load_knowledge_base(self, file_path: str = "data/clinical_respiratory_kb.json") -> dict:
        """Loads the aggregated domain knowledge base JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at {file_path}. Run extraction pipeline first.")
    
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
