import itertools
from typing import Dict, List, Tuple
import networkx as nx
from src.engine.build_dag import CausalGraphBuilder
from data.SymptomsDict import SYMPTOM_ALIASES


def clean_observations(raw_obs: Dict[str, bool], graph: nx.DiGraph) -> Dict[str, bool]:
    """
    Functional mapper to translate colloquial symptom descriptions into canonical graph node IDs using a static alias dictionary.
    """
    cleaned = {}
    for symptom, present in raw_obs.items():
        key = symptom.lower().strip()
        canonical_id = SYMPTOM_ALIASES.get(key, key.replace(" ", "_"))
        # Verify if the node actually exists in the graph before adding
        if canonical_id in graph:
            cleaned[canonical_id] = present
        else:
            print(f"⚠️ Warning: Symptom '{symptom}' (mapped to '{canonical_id}') not found in graph nodes. Skipping.")
    return cleaned


class CausalBayesianInference:
    def __init__(self, graph: nx.DiGraph):
        """Initializes the engine with a pre-constructed Causal Graph."""
        self.graph = graph


    def _get_node_likelihood(self, child_id: str, active_diseases: Dict[str, bool]) -> float:
        """Calculates P(Symptom=1 | Joint Disease State) using Noisy-OR."""
        p0 = self.graph.nodes[child_id].get('leak_probability', 0.02)
        q_prod = 1.0 - p0

        in_edges = self.graph.in_edges(child_id, data=True)
        for parent_disease, _, data in in_edges:
            if active_diseases.get(parent_disease, False):
                p1 = data['p1']
                q_prod *= (1.0 - p1)

        return 1.0 - q_prod

    def infer_joint_posteriors(self, observations: Dict[str, bool]) -> Tuple[Dict[str, float], Dict[Tuple, float]]:
        """
        Evaluates exact joint posterior distribution over the 2^K power set of disease states.
        Returns:
            - marginal_posteriors: P(Disease_i = True | Observations) for each disease
            - joint_posteriors: Top probable global state combinations
        """
        diseases = [n for n, d in self.graph.nodes(data=True) if d['node_type'] == 'disease']
        num_diseases = len(diseases)
        
        joint_unnormalized = {}

        # Iterate through all 2^K possible combinations of disease states
        for state_tuple in itertools.product([False, True], repeat=num_diseases):
            state_dict = dict(zip(diseases, state_tuple))
            
            # 1. Calculate Joint Prior P(D_1, D_2, ..., D_K)
            prior = 1.0
            for d in diseases:
                p_d = self.graph.nodes[d]['prior']
                prior *= p_d if state_dict[d] else (1.0 - p_d)

            # 2. Calculate Joint Likelihood P(Observed Symptoms | State)
            likelihood = 1.0
            for obs_symptom, obs_value in observations.items():
                if obs_symptom in self.graph:
                    p_obs_true = self._get_node_likelihood(obs_symptom, state_dict)
                    likelihood *= p_obs_true if obs_value else (1.0 - p_obs_true)

            joint_unnormalized[state_tuple] = prior * likelihood

        # Normalize across all 2^K joint states
        total_mass = sum(joint_unnormalized.values())
        if total_mass == 0:
            joint_posteriors = {s: 1.0 / len(joint_unnormalized) for s in joint_unnormalized}
        else:
            joint_posteriors = {s: p / total_mass for s, p in joint_unnormalized.items()}

        # 3. Compute Marginal Posteriors for each individual disease P(D_i = True | Obs)
        marginal_posteriors = {}
        for idx, disease in enumerate(diseases):
            prob_true = sum(p for s, p in joint_posteriors.items() if s[idx] is True)
            marginal_posteriors[disease] = prob_true

        # Also compute explicit "Healthy" probability (State where all diseases are False)
        all_false_tuple = tuple([False] * num_diseases)
        marginal_posteriors["[Healthy / No Modeled Disease]"] = joint_posteriors.get(all_false_tuple, 0.0)

        return marginal_posteriors, joint_posteriors


if __name__ == "__main__":

    builder = CausalGraphBuilder()
    kb_data = builder.load_knowledge_base("data/clinical_respiratory_kb.json")
    
    dag_engine = builder.build_from_knowledge_base(kb_data)
    inf_engine = CausalBayesianInference(dag_engine)

    print("✅ Joint Multi-Disease DAG Inference Engine built successfully!\n")

    # TEST CASE: Using natural, unformatted clinical descriptions
    raw_obs = {
        "Fever": True,
        "Loss of Taste": True,
        "Shortness of Breath": True,
        "Cough": True                  
    }

    print("--- Raw Patient Observations ---")
    for symptom, present in raw_obs.items():
        status = "Present (True)" if present else "Absent (False)"
        print(f"  • {symptom}: {status}")
    print()

    # Functional mapping clean-up
    cleaned_obs = clean_observations(raw_obs, inf_engine.graph)

    print("--- Mapped Canonical Observations ---")
    for symptom, present in cleaned_obs.items():
        print(f"  • Canonical ID: '{symptom}' -> {present}")
    print()

    # Causal inference
    marginals, joints = inf_engine.infer_joint_posteriors(cleaned_obs)

    print("--- Marginal Disease Posteriors P(D_i = True | Obs) ---")
    for d, p in marginals.items():
        print(f"  {d}: {p:.2%}")

    print("\n--- Top 3 Most Likely Joint Global States (Power Set) ---")
    diseases = [n for n, d in inf_engine.graph.nodes(data=True) if d['node_type'] == 'disease']
    sorted_joints = sorted(joints.items(), key=lambda x: x[1], reverse=True)[:3]
    
    for state_tuple, prob in sorted_joints:
        active = [diseases[i] for i, flag in enumerate(state_tuple) if flag]
        state_str = " + ".join(active) if active else "Healthy"
        print(f"  State [{state_str}]: {prob:.2%}")

