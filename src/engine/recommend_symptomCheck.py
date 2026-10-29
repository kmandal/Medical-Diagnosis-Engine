import math
from typing import Dict, List, Tuple
import networkx as nx
from src.engine.build_dag import CausalGraphBuilder
from src.engine.BayesCausalInf_dag import CausalBayesianInference, clean_observations

class DiagnosticSymptomRecommender:
    def __init__(self, engine: CausalBayesianInference):
        self.engine = engine

    def calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculates Shannon Entropy in bits over disease posteriors."""
        return -sum(p * math.log2(p) for p in probabilities.values() if p > 0)

    def recommend_next_diagnostic_symptom_test(
        self, 
        current_posteriors: Dict[str, float], 
        current_observations: Dict[str, bool],
        candidate_tests: List[str]
    ) -> Tuple[str, float]:
        """
        Selects the candidate test symptom that maximizes Expected Information Gain (EIG) using Shannon entropy reduction.
        """
        # Filter posteriors to only include actual disease nodes (exclude healthy states)
        disease_posteriors = {
            d: p for d, p in current_posteriors.items() 
            if not d.startswith("[") and self.engine.graph.nodes.get(d, {}).get("node_type") == "disease"
        }
        
        if not disease_posteriors:
            return None, 0.0

        base_entropy = self.calculate_entropy(disease_posteriors)
        best_test = None
        max_gain = -1.0
        
        for test in candidate_tests:
            # Normalize or map candidate symptom test names
            test_id = test.lower().strip().replace(" ", "_")
            if test_id not in self.engine.graph:
                continue

            # Skip tests that have already been observed
            if test_id in current_observations:
                continue

            # Evaluate P(Test=True) across current beliefs
            p_test_true = 0.0
            for disease, p_d in disease_posteriors.items():
                # Get likelihood of this symptom given the specific active disease
                active_state = {disease: True}
                p_s_given_d = self.engine._get_node_likelihood(test_id, active_state)
                p_test_true += p_s_given_d * p_d
            
            p_test_false = 1.0 - p_test_true

            # Prevent division or zero probability issues
            if p_test_true <= 0.0 or p_test_false <= 0.0:
                continue

            # Posteriors if Test symptom = True
            obs_true = {**current_observations, test_id: True}
            post_true_raw, _ = self.engine.infer_joint_posteriors(obs_true)
            post_true_diseases = {
                d: p for d, p in post_true_raw.items() 
                if not d.startswith("[")
            }

            # Posteriors if Test symptom = False
            obs_false = {**current_observations, test_id: False}
            post_false_raw, _ = self.engine.infer_joint_posteriors(obs_false)
            post_false_diseases = {
                d: p for d, p in post_false_raw.items() 
                if not d.startswith("[")
            }

            # Expected Entropy after observing the test result
            exp_entropy = (p_test_true * self.calculate_entropy(post_true_diseases)) + \
                          (p_test_false * self.calculate_entropy(post_false_diseases))
            
            info_gain = base_entropy - exp_entropy

            if info_gain > max_gain:
                max_gain = info_gain
                best_test = test_id

        return best_test, max_gain


if __name__ == "__main__":
    # --- TEST RUNNER ---
    builder = CausalGraphBuilder()
    kb_data = builder.load_knowledge_base("data/clinical_respiratory_kb.json")
    dag_graph = builder.build_from_knowledge_base(kb_data)
    
    inf_engine = CausalBayesianInference(dag_graph)
    recommender = DiagnosticSymptomRecommender(inf_engine)

    # Partial patient observations (e.g., patient came in with fever and cough, but loss of taste hasn't been checked yet)
    initial_obs = {"fever": True, "cough": True}
    cleaned_obs = clean_observations(initial_obs, inf_engine.graph)

    # Get current marginal posteriors
    marginals, _ = inf_engine.infer_joint_posteriors(cleaned_obs)

    print("--- Current Clinical Context ---")
    print(f"Observations so far: {list(cleaned_obs.keys())}")
    print("\nInfered Diseases:")
    for d, p in marginals.items():
        if not d.startswith("["):
            print(f"  • {d}: {p:.2%}")

    # Candidate tests/symptoms to evaluate for next best information gain
    candidate_tests = ["anosmia", "ageusia", "dyspnea", "myalgia", "chills"]

    best_test, gain = recommender.recommend_next_diagnostic_symptom_test(marginals, cleaned_obs, candidate_tests)

    print("\n--- Active Diagnostic Recommendation ---")
    if best_test:
        print(f"💡 Recommended Next Symptom to Check: '{best_test.upper()}'")
        print(f"📈 Expected Information Gain (Entropy Reduction): {gain:.4f} bits")
    else:
        print("⚠️ No candidate tests provided or available to reduce entropy.")
