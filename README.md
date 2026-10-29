# Medical-Diagnosis-Engine

An autonomous, agentic medical knowledge extraction and causal probabilistic decision-support system. 

`Medical-Diagnosis-Engine` bridges live literature extraction with rigorous probabilistic graphical modeling. It autonomously scrapes clinical literature via Tavily, extracts structured parameters using Pydantic-validated LLM agents, aggregates modular domain knowledge bases, and performs exact multi-disease power-set Bayesian inference coupled with Shannon Entropy-based Expected Information Gain (EIG) test recommendations.

---

## 📂 Repository Architecture

```text
Medical-Diagnosis-Engine/
├── data/
│   ├── queries.json             # Batch target diseases and symptoms config
│   ├── SymptomsDict.py          # Static canonical entity mapping dictionary
│   ├── clinical_respiratory_kb.json      # Aggregated domain knowledge base (Generated)
│   └── clinical_respiratory_raw.json     # Aggregated raw literature extractions (Generated)
├── src/
│   ├── agents/                  # LangGraph multi-agent extraction pipeline
│   │   ├── nodes.py             # Parser, Translator, Critic, and Storage nodes
│   │   ├── prompts.py           # System prompts and strict extraction guardrails
│   │   ├── state.py             # LangGraph state typed dictionary definition
│   │   └── workflow.py          # Orchestration graph and batch runner
│   └── engine/                  # Probabilistic reasoning and decision support
│       ├── build_dag.py         # Dynamic NetworkX Causal Graph builder
│       ├── BayesCausalInf_dag.py # Exact 2^K power-set joint multi-disease inference
│       └── recommend_symptomCheck.py # Shannon Entropy & EIG active symptom recommender
├── .env.example                 # Environment variables template
├── .gitignore
├── requirements.txt             # Python dependencies
└── README.md

```

🚀 Core Features

*  Autonomous Agentic Ingestion: Leverages LangGraph and Tavily to fetch real-time clinical studies and literature snippets based on external query targets.
*  Deterministic Validation: Enforces strict Pydantic parsing schemas and iterative critic validation loops to guarantee mathematical consistency ($0 < P < 1$, $p_1 > p_0$).
*  Domain-Level Aggregation: Implements an upsert persistence pattern, grouping multi-disease extractions into modular domain files (e.g., respiratory).
*  Exact Power-Set Causal Inference: Evaluates co-infections and multi-morbidity across the $2^K$ joint disease state space using a Noisy-OR likelihood model.
*  Active Clinical Decision Support: Calculates Shannon Entropy and Expected Information Gain (EIG) to recommend unobserved symptoms that minimize diagnostic uncertainty the fastest.


🛠️ Installation & Setup

1. Clone the Repository
```
git clone [https://github.com/kmandal/Medical-Diagnosis-Engine.git](https://github.com/kmandal/Medical-Diagnosis-Engine.git)
cd Medical-Diagnosis-Engine
```

2. Create and Activate Virtual Environment
```
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install Dependencies
```
pip install -r requirements.txt
```

4. Configure Environment Variables

Copy the example environment file and add your API keys:
```
cp .env.example .env
```

🧪 Usage
Step 1: Run the Agentic Extraction Pipeline

Reads target queries from data/queries.json, scrapes the web, validates outputs, and updates the domain knowledge base (data/clinical_respiratory_kb.json):
```
python -m src.agents.workflow
```

Step 2: Run Causal Bayesian Inference

Converts the domain knowledge base into a dynamic NetworkX Directed Acyclic Graph (DAG) and evaluates multi-disease posteriors given patient observations:
```
python -m src.engine.BayesCausalInf_dag
```

Step 3: Run Active Symptom Check Recommender

Calculates entropy reduction and Expected Information Gain (EIG) to suggest the next best clinical check:
```
python -m src.engine.recommend_symptomCheck
```

📜 License

This project is open-source under the MIT License.
