## Pipeline – Kurzanleitung (komplett)

```bash
# =========================
# 1) Setup
# =========================
python -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\activate             # Windows
pip install -r requirements.txt

# =========================
# 2) Environment (LLM optional)
# =========================
export ALLOW_LLM_CALLS=1             # nur falls LLM genutzt wird
export OPENAI_API_KEY="..."          # optional
export GEMINI_API_KEY="..."          # optional

# Lokales Modell (optional)
ollama serve
ollama pull mistral

# =========================
# 3) Pipeline (MVP)
# =========================
# 3.1 Crawling
python -m src.cli crawl-mvp \
  --dataset configs/datasets/mvp_bayern.yaml \
  --out data/de_by/staging/mvp

# 3.2 Extraktion (HTML/PDF → Text)
python -m src.cli extract-documents \
  --raw data/de_by/staging/mvp/documents_raw.ndjson \
  --out data/de_by/staging/mvp/documents_extracted.ndjson

# 3.3 Candidate-Building
python -m src.cli extract-policies \
  --extracted data/de_by/staging/mvp/documents_extracted.ndjson \
  --out data/de_by/staging/mvp/policy_candidates.ndjson

# 3.4 LLM-Gate / Klassifikation (optional)
python -m src.cli auto-label-policies \
  --candidates data/de_by/staging/mvp/policy_candidates.ndjson \
  --out data/de_by/staging/mvp/measures_llm.ndjson \
  --llm-config configs/llm.yml \
  --profile measure_gate_closed_taxonomy_ollama \
  --max 200

# =========================
# 4) Evaluation & Abbildungen
# =========================
python src/eval_submission/llm_gate_eval.py

python src/eval_submission/postprocess_gate.py \
  --pred_csv <PRED_CSV> \
  --gold_csv data/de_by/goldstandard/goldstandard_gold.csv \
  --outdir <OUTDIR> \
  --text_col text_for_annotation \
  --pos_conf 0.85 \
  --abstain_conf 0.60

# Thesis-Figuren (matplotlib)
python generate_results_figures.py
