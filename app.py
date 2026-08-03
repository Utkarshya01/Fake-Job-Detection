import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

from src.dataset_loader import load_job_dataset
from src.preprocessing import preprocess_job_data

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_pipeline.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")
EXPERIMENTS_PATH = os.path.join(MODELS_DIR, "experiments_summary.json")
INDICATORS_PATH = os.path.join(MODELS_DIR, "feature_indicators.json")

# Known high-risk scam triggers for rule-based interpretability highlight
SCAM_TRIGGER_WORDS = [
    "wire transfer", "western union", "moneygram", "bitcoin", "crypto", "cashier check",
    "processing fee", "starter kit", "envelope stuffing", "home processor", "no experience needed",
    "earn 5000", "earn 1000", "immediate hire", "bank account", "direct deposit", "financial coordinator",
    "package dispatch", "mystery shopper", "personal assistant", "wire funds", "cash check"
]

# Global cache for dataset and model
DATASET_DF = None
PIPELINE = None

def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        if os.path.exists(MODEL_PATH):
            PIPELINE = joblib.load(MODEL_PATH)
        else:
            # Fallback training trigger if model missing
            from train import run_training_pipeline
            run_training_pipeline()
            PIPELINE = joblib.load(MODEL_PATH)
    return PIPELINE

def get_dataset():
    global DATASET_DF
    if DATASET_DF is None:
        DATASET_DF = load_job_dataset()
    return DATASET_DF


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    model_loaded = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "online",
        "model_loaded": model_loaded,
        "message": "Fake Job Posting Detection API active."
    })


@app.route("/api/eda-stats", methods=["GET"])
def eda_stats():
    df = get_dataset()
    
    total = int(len(df))
    fraud_count = int(df['fraudulent'].sum())
    real_count = total - fraud_count

    # Missing profile breakdown
    no_profile_fraud = int(((df['fraudulent'] == 1) & (df['company_profile'].fillna('').str.strip() == '')).sum())
    no_profile_real = int(((df['fraudulent'] == 0) & (df['company_profile'].fillna('').str.strip() == '')).sum())

    no_logo_fraud = int(((df['fraudulent'] == 1) & (df['has_company_logo'] == 0)).sum())
    no_logo_real = int(((df['fraudulent'] == 0) & (df['has_company_logo'] == 0)).sum())

    # Top industries for fraud
    industry_counts = df[df['fraudulent'] == 1]['industry'].value_counts().head(5).to_dict()

    return jsonify({
        "total_postings": total,
        "class_distribution": {
            "legitimate": real_count,
            "fraudulent": fraud_count,
            "fraud_percentage": round((fraud_count / total) * 100, 2)
        },
        "missing_features_analysis": {
            "missing_company_profile": {
                "fraudulent_lacking_profile": no_profile_fraud,
                "legitimate_lacking_profile": no_profile_real
            },
            "missing_company_logo": {
                "fraudulent_lacking_logo": no_logo_fraud,
                "legitimate_lacking_logo": no_logo_real
            }
        },
        "top_fraudulent_industries": industry_counts
    })


@app.route("/api/model-performance", methods=["GET"])
def model_performance():
    metrics = {}
    experiments = {}

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    if os.path.exists(EXPERIMENTS_PATH):
        with open(EXPERIMENTS_PATH, "r") as f:
            experiments = json.load(f)

    return jsonify({
        "models": metrics,
        "experiments": experiments
    })


@app.route("/api/feature-indicators", methods=["GET"])
def feature_indicators():
    indicators = {}
    if os.path.exists(INDICATORS_PATH):
        with open(INDICATORS_PATH, "r") as f:
            indicators = json.load(f)
    else:
        indicators = {
            "top_fraudulent_indicators": ["wire transfer", "cash payment", "immediate start", "no interview", "bank deposit"],
            "top_legitimate_indicators": ["bachelor degree", "full time", "experience required", "company benefits", "engineering"]
        }
    return jsonify(indicators)


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        pipeline = get_pipeline()
        data = request.get_json(force=True) or {}

        title = str(data.get("title", ""))
        company_profile = str(data.get("company_profile", ""))
        description = str(data.get("description", ""))
        requirements = str(data.get("requirements", ""))
        benefits = str(data.get("benefits", ""))
        salary_range = str(data.get("salary_range", ""))

        telecommuting = int(data.get("telecommuting", 0))
        has_company_logo = int(data.get("has_company_logo", 1))
        has_questions = int(data.get("has_questions", 1))

        # Create single row DataFrame
        input_dict = {
            "job_id": [1],
            "title": [title],
            "company_profile": [company_profile],
            "description": [description],
            "requirements": [requirements],
            "benefits": [benefits],
            "salary_range": [salary_range],
            "telecommuting": [telecommuting],
            "has_company_logo": [has_company_logo],
            "has_questions": [has_questions]
        }
        df_input = pd.DataFrame(input_dict)
        df_processed = preprocess_job_data(df_input, clean_text_flag=True)

        # Model Inference
        pred_label = int(pipeline.predict(df_processed)[0])
        
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df_processed)[0]
            fraud_prob = float(probs[1])
        else:
            fraud_prob = 0.95 if pred_label == 1 else 0.05

        risk_score_percent = round(fraud_prob * 100, 2)
        status = "FRAUDULENT" if pred_label == 1 else "LEGITIMATE"

        # Highlight suspicious words
        full_text = f"{title} {company_profile} {description} {requirements} {benefits}".lower()
        detected_triggers = [word for word in SCAM_TRIGGER_WORDS if word in full_text]

        # Key risk factors list
        risk_factors = []
        if company_profile.strip() == "":
            risk_factors.append("Missing Company Profile background information.")
        if has_company_logo == 0:
            risk_factors.append("No official company logo uploaded.")
        if len(detected_triggers) > 0:
            risk_factors.append(f"Contains high-risk financial/scam terms: {', '.join(detected_triggers)}")
        if telecommuting == 1 and ("entry level" in full_text or "no experience" in full_text) and salary_range:
            risk_factors.append("High salary advertised for remote entry-level position without experience.")

        if len(risk_factors) == 0 and pred_label == 0:
            risk_factors.append("Posting contains legitimate company background, clear technical requirements, and standard structure.")

        return jsonify({
            "status": status,
            "is_fraudulent": bool(pred_label == 1),
            "fraud_probability": round(fraud_prob, 4),
            "risk_score_percent": risk_score_percent,
            "risk_level": "HIGH RISK" if fraud_prob >= 0.6 else ("MEDIUM RISK" if fraud_prob >= 0.35 else "LOW RISK"),
            "detected_triggers": detected_triggers,
            "risk_factors": risk_factors
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


@app.route("/api/batch-predict", methods=["POST"])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded in request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Selected file is empty"}), 400

        # Robust CSV decoding
        try:
            df_upload = pd.read_csv(file, encoding='utf-8')
        except Exception:
            file.seek(0)
            df_upload = pd.read_csv(file, encoding='latin1')

        if len(df_upload) == 0:
            return jsonify({"error": "Uploaded CSV file contains no data rows"}), 400

        pipeline = get_pipeline()
        df_processed = preprocess_job_data(df_upload, clean_text_flag=True)

        preds = pipeline.predict(df_processed)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(df_processed)[:, 1]
        else:
            probs = [0.95 if p == 1 else 0.05 for p in preds]

        results = []
        for idx, row in df_upload.iterrows():
            fraud_prob = float(probs[idx])
            is_fraud = int(preds[idx])
            results.append({
                "index": idx + 1,
                "title": str(row.get("title", "N/A")),
                "prediction": "FRAUDULENT" if is_fraud == 1 else "LEGITIMATE",
                "fraud_probability": round(fraud_prob, 4),
                "risk_score": round(fraud_prob * 100, 1)
            })

        total_scanned = len(results)
        total_fraud = sum(1 for r in results if r["prediction"] == "FRAUDULENT")

        return jsonify({
            "total_scanned": total_scanned,
            "flagged_fraudulent": total_fraud,
            "fraud_rate_percent": round((total_fraud / max(total_scanned, 1)) * 100, 2),
            "results": results[:50]  # Return top 50 rows preview
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("[Server] Starting Fake Job Posting Detection Web Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
