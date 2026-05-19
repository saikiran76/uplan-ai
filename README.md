## ⚠️ Scope & Current Limitations (Proof of Concept)

<img width="1851" height="857" alt="image" src="https://github.com/user-attachments/assets/6aefce06-66ea-4003-ab64-99747de68982" />


## Quick look into tech stack

| Component | Technology | Description / Role |
| :--- | :--- | :--- |
| **Backend** | Python (FastAPI) | High-performance async API framework for pipeline orchestration and background tasks. |
| **AI / Core Engine** | Gemini | Powering the LLM orchestration agents, document parsing, and extraction logic. |
| **Frontend** | HTML / CSS / JS | Lightweight, responsive interface for document uploads and real-time status tracking. |
| **Deployment** | Streamlit | Rapid prototyping interface tool used for initial internal UI/UX evaluation. |
| **Hosting** | Hugging Face Spaces | Cloud hosting infrastructure leveraging Git-based workflows to run the live POC pipeline. |


AS A NOTE,

As a brief addendum to our submission video, we want to be transparent about the current boundaries of this prototype.

* **Data Scarcity & Prompt Streamlining:** The current LLM agents and system prompts are highly optimized for a specific subset of visa categories based on the limited data we had access to during the hackathon.
* **Rule Engine Scope:** The deterministic algebraic rule engine currently evaluates a constrained set of financial and temporal thresholds.
* **The Takeaway:** This is a localized Proof-of-Concept (POC). It will likely miss nuances or hallucinate if tested with highly obscure visa types or extreme edge-case dossier structures that fall outside our current training/prompting distribution.

## 🚀 Road to Production

To scale this from a hackathon POC to an enterprise-grade universal adjudication engine, our next phases include:

* **True Fine-Tuning Lifecycle:** Transitioning from heavy prompt-engineering to a rigorous ML pipeline. We will acquire diverse, global dossier data, establish strict train/test/validation splits, and fine-tune the extraction models.
* **Local Compute & Absolute Privacy:** Moving away from provider APIs to training and hosting our own foundational models on private, air-gapped infrastructure. This guarantees zero-data-leakage for highly sensitive PII and financial records.
* **LoRA / QLoRA Optimization:** Utilizing Parameter-Efficient Fine-Tuning (PEFT) to rapidly spin up specialized models for different countries/jurisdictions, balancing high accuracy with compute-cost trade-offs.
* **Enterprise Infrastructure:** Scaling the FastAPI background-task architecture with message queues (e.g., Celery/Redis), load-balanced GPU worker nodes, and robust OAuth2/JWT authentication.

## Test Data for Evaluators

Because we couldn't showcase the raw PII data in our submission video, we are providing the exact test dossiers we used to evaluate the pipeline.

* **Sample identity document**
<img width="1251" height="1848" alt="Screenshot_2026-05-19-10-39-26-26_e2d5b3f32b79de1d45acd1fad96fbb0f" src="https://github.com/user-attachments/assets/f5f73364-c012-4c5d-87b0-30f770dd1f00" />

<img width="2459" height="1979" alt="Screenshot_2026-05-19-10-35-22-26_e2d5b3f32b79de1d45acd1fad96fbb0f" src="https://github.com/user-attachments/assets/b168ac5e-937a-4068-9e7d-40a1a1f38914" />
* **Bank records**

<img width="1727" height="1287" alt="Screenshot_2026-05-19-10-33-35-62_a1b1bbe5f63d5b96c1a0f87c197ebfae" src="https://github.com/user-attachments/assets/790e1103-4ae6-4116-970f-374bd97f85c9" />
* **Sample affadavit(one legal doc)**


Want to run the pipeline yourself? We have packaged the test documents into a zip file so you can upload them to the live Hugging Face Space and watch the async architecture in real-time



[👉 Download the Test Dossier Data here (Google Drive)](#) or simply look at the test_files folder in the repo for testing.

