## ⚠️ Scope & Current Limitations (Proof of Concept)

As a brief addendum to our submission video, we want to be transparent about the current boundaries of this prototype.

* **Data Scarcity & Prompt Streamlining:** The current LLM agents and system prompts are highly optimized for a specific subset of visa categories based on the limited data we had access to during the hackathon.
* **Rule Engine Scope:** The deterministic algebraic rule engine currently evaluates a constrained set of financial and temporal thresholds.
* **The Takeaway:** This is a localized Proof-of-Concept (POC). It will likely miss nuances or hallucinate if tested with highly obscure visa types or extreme edge-case dossier structures that fall outside our current training/prompting distribution.

## 🚀 The Roadmap to Production

To scale this from a hackathon POC to an enterprise-grade universal adjudication engine, our next phases include:

* **True Fine-Tuning Lifecycle:** Transitioning from heavy prompt-engineering to a rigorous ML pipeline. We will acquire diverse, global dossier data, establish strict train/test/validation splits, and fine-tune the extraction models.
* **Local Compute & Absolute Privacy:** Moving away from provider APIs to training and hosting our own foundational models on private, air-gapped infrastructure. This guarantees zero-data-leakage for highly sensitive PII and financial records.
* **LoRA / QLoRA Optimization:** Utilizing Parameter-Efficient Fine-Tuning (PEFT) to rapidly spin up specialized models for different countries/jurisdictions, balancing high accuracy with compute-cost trade-offs.
* **Enterprise Infrastructure:** Scaling the FastAPI background-task architecture with message queues (e.g., Celery/Redis), load-balanced GPU worker nodes, and robust OAuth2/JWT authentication.

## 🧪 Test Data for Evaluators

Because we couldn't showcase the raw PII data in our submission video, we are providing the exact test dossiers we used to evaluate the pipeline.

*(Include 2-3 screenshots of your blurred/mocked test documents here)*


Want to run the pipeline yourself? We have packaged the test documents into a zip file so you can upload them to the live Hugging Face Space and watch the async architecture in real-time.

[👉 Download the Test Dossier Data (Google Drive)](#)

