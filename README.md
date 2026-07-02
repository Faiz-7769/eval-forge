# ⚡ LLM Eval Forge

> A multi-metric benchmarking framework that evaluates local and cloud LLMs on latency, GPU utilization, faithfulness, and answer relevancy — before choosing one for production.

---

## 🧠 Why This Project Exists

Before building a RAG system or any transformer-based use case, companies need to answer one question: **which model should I actually use?**

Instead of guessing, I built a structured evaluation pipeline that benchmarks three LLMs across hardware performance, output quality, and instruction-following ability — using an independent judge model to avoid self-preference bias in scoring.

This framework helps identify which model to use under which conditions.

And many college students can't afford the api credits and they still want to use LLMs then this also helps students choose between running models locally and using free cloud API providers for their projects.

During development, several other local and cloud models (additional DeepSeek, Groq-llama, Mistral and gemini configurations) were also benchmarked before narrowing down to the final three reported below — the extra CSVs in `evaluations/` and `metrics/` are artifacts of that exploration phase, kept for transparency.

---

## 🏗️ Architecture

```
Phase 1 — Inference (Google Colab)
  ├── Microsoft Phi-4 14B (4-bit quantized, local on T4 GPU)
  ├── Qwen3-32B via Groq API
  └── Mistral-Small-2506 via Mistral API
          ↓
    Metrics collected per question:
    prompt_tokens, output_tokens, latency_sec,
    tokens_per_sec, gpu_usage_pct, gpu_temp_c, gpu_mem_used_mb

Phase 2 — Evaluation (Local)
  └── DeepEval (AnswerRelevancy + FaithfulnessMetric)
      judged by Gemini 3.1 Flash Lite (independent — not one of the benchmark models)
          ↓
    Scores: answer_relevancy, faithfulness
    Split by question_type: factual | constraint

Phase 3 — Storage
  └── SQLite (benchmark.db) — single table, all 90 rows

Phase 4 — Dashboard
  └── Streamlit — one-page scroll, interactive charts
```

---

## 📊 Results Summary

| Model | Avg Latency | Tokens/sec | Answer Relevancy | Faithfulness |
|---|---|---|---|---|
| Phi-4 14B (Local) | 36.77s | 6.3 | 0.919 | **0.968** |
| Qwen3-32B (Groq) | 3.30s | **423.9** | **0.920** | 0.914 |
| Mistral-Small (API) | **2.93s** | 129.7 | 0.904 | 0.966 |

**Key findings:**
- Phi-4 achieves the highest faithfulness (0.968) despite running locally on quantized weights — competitive with cloud models on quality
- Qwen3-32B has 67x higher throughput than Phi-4 at a fraction of the latency
- All three models show a consistent drop in answer relevancy on constraint-following tasks vs factual Q&A — regardless of model size or deployment method
- Mistral wins on raw latency, Qwen3 wins on throughput, Phi-4 wins on faithfulness

---

## 🗂️ Benchmark Dataset

30 hand-curated questions across 3 categories and 3 difficulty levels:

| Category | Count | Difficulty range |
|---|---|---|
| Logical Reasoning & Math | 10 | Easy → Hard |
| Coding & Syntax | 10 | Easy → Hard |
| Instruction Following & Constraints | 10 | Easy → Hard |

Questions were designed so that constraint-following tasks have no single correct answer (open-ended), while factual questions have a verifiable expected answer. This distinction is respected in evaluation — Faithfulness is only scored on factual questions to avoid penalizing valid but different answers on open-ended tasks.

Download the full benchmark dataset from the dashboard or directly from `data/benchmark.json`.

---

## 🔬 Evaluation Methodology

**Why Gemini as judge and not one of the benchmark models?**

Using any of the three benchmark models as the judge would create self-preference bias — models tend to score answers similar to their own style higher. Gemini 3.1 Flash Lite was chosen as an independent third-party evaluator with no stake in the results.

**Why split factual vs constraint questions for Faithfulness?**

`FaithfulnessMetric` compares the model's answer against a ground truth context. For constraint-following questions (e.g. "write a paragraph without using the letter 'e'"), there is no single correct answer — only rules to follow. Scoring faithfulness against a placeholder ground truth would produce meaningless numbers. These rows have `faithfulness = NULL` by design and are excluded from faithfulness aggregations.

**Why only 30 questions?**

Gemini 3.1 Flash Lite (the judge model) has a 500-request/day quota on the free tier. DeepEval's metrics aren't a single judge call each — `AnswerRelevancyMetric` and `FaithfulnessMetric` internally decompose each answer into statements/claims and generate a verdict per claim before computing a final score, so one metric evaluation costs several judge calls under the hood, not one.

Across 30 questions × 3 models × 2 metrics, this added up to roughly 400 Gemini calls in practice — close to the full quota once reruns for failed/timed-out calls are factored in. The dataset was sized to fit inside that budget rather than being an arbitrary round number, which mirrors a real constraint teams hit when running LLM-as-judge evals at scale: judge quality/cost trades off directly against how many samples you can afford to score.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Local inference | HuggingFace Transformers + BitsAndBytes 4-bit quantization |
| GPU monitoring | `nvidia-ml-py` (continuous polling in background thread) |
| Cloud inference | LangChain + Groq API + Mistral API |
| Evaluation | DeepEval (AnswerRelevancyMetric, FaithfulnessMetric) |
| Judge model | Gemini 3.1 Flash Lite via LangChain Google GenAI |
| Storage | SQLite |
| Dashboard | Streamlit + Plotly |

---

## 📁 Project Structure

```
eval-forge/
├── notebooks/
│   ├── final_run_v2.ipynb     # canonical run: inference + GPU metrics + DeepEval scoring
│   └── evaluation_v1.ipynb, final_run.ipynb, final_run_v1.ipynb, normal_run_final.ipynb
│                              # earlier iterations from model exploration, kept for reference
├── data/
│   └── benchmark.json         # 30 curated questions + expected answers
├── evaluations/               # per-model eval CSVs
├── metrics/                   # per-model metrics CSVs
├── benchmark.db               # SQLite — all 90 rows combined
├── combined.csv               # flat export of benchmark.db
├── app.py                     # Streamlit dashboard
└── requirements.txt
```

> `notebooks/final_run.ipynb` is the canonical run behind the results reported below. The other notebooks are earlier iterations from the model-exploration phase (see note above and [Roadmap](#-roadmap)).

---

## 🚀 Running the Dashboard

```bash
# Clone the repo
git clone https://github.com/yourusername/eval-forge
cd eval-forge

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard reads from `benchmark.db` (SQLite) or falls back to CSV files in `evaluations/` if the database isn't present.

---

## 📦 Requirements

```
streamlit
plotly
pandas
deepeval
langchain-google-genai
langchain-mistralai
langchain-groq
transformers
bitsandbytes
accelerate
nvidia-ml-py
python-dotenv
```
Use torch with cuda compatibility if you would like to run the llm locally using huggingface.
Download from official Pytorch website.
---

## 🔑 Environment Variables

Create a `.env` file in the root:

```env
GROQ_API_KEY=your_groq_key
MISTRAL_API_KEY=your_mistral_key
GEMINI_API_KEY=your_gemini_key
HF_TOKEN = your_token
```

---

## ⚙️ Known Limitations & Next Optimizations

- **Sample size:** 30 questions per model is enough to reveal directional trends (latency, throughput, faithfulness gaps) but too small for statistically rigorous claims. Scaling this up is bottlenecked by the judge model's rate limit (see [Evaluation Methodology](#-evaluation-methodology)), not by the pipeline itself — the code already supports arbitrarily larger question sets.
- **Judge quota ceiling:** Gemini 3.1 Flash Lite's 500 req/day free-tier quota caps how many questions × models × metrics can be scored per day. A batched/async judge-call pattern (or a paid tier) would remove this ceiling and let the benchmark scale to hundreds of questions.
- **Notebook sprawl:** Several exploratory notebooks (`evaluation_v1.ipynb`, `final_run_v2.ipynb`, `final_run_v1.ipynb`, `normal_run_final.ipynb`) are kept alongside the canonical `final_run.ipynb`. Next cleanup pass: consolidate into a single parameterized notebook (or script) with model choice as a config flag, and archive/remove the rest.
- **Secrets hygiene:** `.env` is git-ignored, but committed history should be double-checked before making the repo public to confirm no real keys were ever pushed.
- **Model coverage:** Only one model per deployment mode (local/Groq/Mistral) is reported. The exploratory CSVs in `evaluations/` and `metrics/` cover more candidates and could be folded into the dashboard as optional comparisons rather than left as raw files.

---

## 🗺️ Roadmap

### ✅ Phase 1 — Benchmark (Complete)
Multi-metric evaluation of Phi-4 (local), Qwen3-32B (Groq), and Mistral-Small
across latency, GPU utilization, faithfulness, and answer relevancy
judged by an independent Gemini evaluator.

### 📋 Phase 2 — Live Arena (Planned)
An interactive query interface where a user can type any question and all three
models respond simultaneously. Each response will be automatically scored by the
Gemini judge for answer relevancy and faithfulness — displayed side by side in
real time, turning the static benchmark into a live evaluation tool.


---


## Author

**Faiz Ahmed**

Computer Science Undergraduate

Interested in:
- Machine Learning
- LLM Engineering
- MLOps
- Generative AI
- AI Systems

---

## 📄 License

MIT — the benchmark dataset and evaluation framework are free to use, adapt, and extend.