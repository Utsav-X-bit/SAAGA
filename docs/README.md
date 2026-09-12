# SAAGA Documentation System

Welcome to the official documentation for **SAAGA: Strategic and Adaptive Attack Generation for Red Teaming of LLMs**.

The documentation has been consolidated into three canonical categories:

---

## 📚 1. High-Level Theoretical & Architectural Design
**[`architecture/THEORETICAL_ARCHITECTURE.md`](architecture/THEORETICAL_ARCHITECTURE.md)**

*Target Audience:* Researchers, architects, and security engineers.

*Key Contents:*
- Adversarial Capture-the-Flag (CTF) problem formulation
- The Sandwich Defense mathematical model ($S = D_{\text{open}} \oplus P_{\text{attack}} \oplus D_{\text{close}}$)
- The ASR validity crisis and verifiable win attribution without an LLM-as-a-judge
- 4-signal deterministic priority ladder and failure mode taxonomy
- Policy vs. wording decoupling (Planner-Generator architecture)
- Structured XML `<plan>` contract and transition dynamics
- 6-layer extraction pipeline theory
- Lifelong learning loop (empirical transition matrices, Thompson sampling, FAISS vector RAG)
- Offensive mutation fallback: inverting JailGuard's defensive mutators (SR, PI, TL, EN)
- End-to-end Mermaid architecture and data flow diagrams

---

## 💻 2. Low-Level Codebase & Implementation Reference
**[`codebase/CODEBASE_REFERENCE.md`](codebase/CODEBASE_REFERENCE.md)**

*Target Audience:* Core maintainers, framework contributors, and code reviewers.

*Key Contents:*
- Exhaustive file-by-file reference covering every module in the repository
- `src/saaga/`: Core data models, provider abstraction layer, adversarial agents, evaluators, fuzzing, memory, reporting, execution engine, and CLI
- `scripts/` & `hpc/`: Dataset tools, training pipelines, analysis tools, and cluster launchers
- `ui/`: Interactive React 18 / Vite / Tailwind evaluation dashboard
- `JailGuard/`: Clean reimplementation (`jailguard_reimpl/`) and perturbation baselines
- `combination/`: Mutation fallback bridge and unit test suites
- `SAAGA/`: Preserved legacy benchmark results archive

---

## 🚀 3. Comprehensive Operational & Usage Guide
**[`usage/USAGE_GUIDE.md`](usage/USAGE_GUIDE.md)**

*Target Audience:* Users, developers, and ML engineers evaluating models.

*Key Contents:*
- Installation and dependency feature sets (`pip install -e ".[all]"`)
- Complete CLI reference:
  - `saaga run`: Single-scenario attack with all options, defaults, and examples
  - `saaga bench`: High-throughput dataset benchmarking and sharding
  - `saaga download-models`: Automated weight and tokenizer fetcher (`tl-mutator`, `base-lora`, etc.)
  - `saaga serve`: Web UI and FastAPI server launch
- Architecture-independent model providers:
  - Locally hosted vLLM / Ollama servers (`--victim-url`)
  - Cloud API providers (OpenAI, DeepSeek, Groq, Together, OpenRouter)
  - In-process GPU cluster vLLM engine
  - Hugging Face Transformers pipeline
- Multi-GPU HPC execution via SLURM (`hpc/saaga_benchmark_4gpu_vllm.sh`)
- Python Programmatic API with runnable code examples
- Troubleshooting and common operational issues

---

## 📊 4. Evaluation Metrics & Mathematical Formulation
**[`EVALUATION_METRICS.md`](EVALUATION_METRICS.md)**

*Target Audience:* Researchers, benchmark authors, and safety evaluation teams.

*Key Contents:*
- The Two-Axis Evaluation Framework: Recoverability Difficulty vs. Model Robustness
- Controlled Difficulty Tiers (`direct`, `deterministic`, `indirect`, `not_recoverable`)
- 4-Signal Deterministic Verification Ladder (`gt_leak`, `access_granted`, `verified_candidate`, `extractor_match`, `none`)
- Severity of Breach ladder ($S_{\text{break}} \in [0.0, 1.0]$)
- Mathematical formulas for all primary scores:
  - Per-Tier Defense Strength Score ($DSS_\tau$)
  - Headline Difficulty-Weighted Defense Strength Score ($DSS_w$)
  - Secret Protection ($SP$) and Compliance Resistance ($CR$) two-axis decomposition
- Auxiliary Metrics: Mean Attempts to Break ($MTB$), Mean Severity of Break ($MSB$), Leak Resistance ($LR$)
- Dynamic search scores: Cooperation Score ($S_{\text{coop}}$) and Mutation Fallback Score ($S_{\text{fallback}}$)
- Diagnostic Failure Mode Classification Taxonomy
- Empirical 3-model benchmark comparison table (Llama-3-8B vs. Mistral-7B vs. Gemma-2B)
