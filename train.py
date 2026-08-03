import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.dataset_loader import load_job_dataset
from src.preprocessing import preprocess_job_data
from src.feature_engineering import build_feature_pipeline
from evaluate import compute_model_metrics, extract_top_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def run_training_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("=" * 60)
    print("      FAKE JOB POSTING DETECTION: TRAINING & EVALUATION      ")
    print("=" * 60)

    # 1. Load Dataset
    raw_df = load_job_dataset()
    print(f"\n[1/5] Dataset Loaded: {len(raw_df)} postings.")
    print(f"      Fraudulent: {raw_df['fraudulent'].sum()} | Legitimate: {(raw_df['fraudulent']==0).sum()}")

    # 2. Preprocess Data
    print("\n[2/5] Preprocessing text and extracting structured features...")
    df = preprocess_job_data(raw_df, clean_text_flag=True)

    # Train/Test Split
    X = df
    y = df['fraudulent'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train Set: {len(X_train)} samples | Test Set: {len(X_test)} samples.")

    # 3. Model Zoo Setup
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SGD Classifier (SVM)": SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42)
    }

    model_results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_pipeline = None

    print("\n[3/5] Training Classifiers on Combined TF-IDF + Metadata Pipeline...")
    
    # Base feature pipeline
    feature_prep = build_feature_pipeline(vectorizer_type='tfidf', max_features=5000)

    for name, clf in models.items():
        print(f"      --> Training {name}...")
        
        pipeline = Pipeline([
            ('preprocessor', build_feature_pipeline(vectorizer_type='tfidf', max_features=5000)),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            decision = pipeline.decision_function(X_test)
            y_prob = (decision - decision.min()) / (decision.max() - decision.min())
        else:
            y_prob = None

        metrics = compute_model_metrics(y_test, y_pred, y_prob)
        model_results[name] = metrics

        print(f"          Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_score']:.4f} | Recall: {metrics['recall']:.4f} | ROC-AUC: {metrics['roc_auc']}")

        if metrics['f1_score'] > best_f1_score:
            best_f1_score = metrics['f1_score']
            best_model_name = name
            best_pipeline = pipeline

    print(f"\n[4/5] Best Model Identified: '{best_model_name}' (F1-Score = {best_f1_score:.4f})")

    # 4. Impact Experiments (Ablation Analysis)
    print("\n[5/5] Conducting Preprocessing & Vectorization Impact Experiments...")
    experiments = {}

    # Exp 1: Raw Text vs Cleaned Text (using Logistic Regression)
    df_raw = preprocess_job_data(raw_df, clean_text_flag=False)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        df_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipe_raw = Pipeline([
        ('preprocessor', build_feature_pipeline(vectorizer_type='tfidf', max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    pipe_raw.fit(X_train_raw, y_train_raw)
    pred_raw = pipe_raw.predict(X_test_raw)
    prob_raw = pipe_raw.predict_proba(X_test_raw)[:, 1]
    metrics_raw = compute_model_metrics(y_test_raw, pred_raw, prob_raw)

    experiments["Raw_vs_Cleaned_Text"] = {
        "Raw_Text": metrics_raw,
        "Cleaned_Text": model_results["Logistic Regression"]
    }

    # Exp 2: TF-IDF vs CountVectorizer
    pipe_count = Pipeline([
        ('preprocessor', build_feature_pipeline(vectorizer_type='count', max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    pipe_count.fit(X_train, y_train)
    pred_count = pipe_count.predict(X_test)
    prob_count = pipe_count.predict_proba(X_test)[:, 1]
    metrics_count = compute_model_metrics(y_test, pred_count, prob_count)

    experiments["TFIDF_vs_CountVectorizer"] = {
        "TFIDF_Vectorizer": model_results["Logistic Regression"],
        "Count_Vectorizer": metrics_count
    }

    # Exp 3: Scaled vs Unscaled Metadata Features
    pipe_unscaled = Pipeline([
        ('preprocessor', build_feature_pipeline(vectorizer_type='tfidf', max_features=5000, use_scaling=False)),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    pipe_unscaled.fit(X_train, y_train)
    pred_unscaled = pipe_unscaled.predict(X_test)
    prob_unscaled = pipe_unscaled.predict_proba(X_test)[:, 1]
    metrics_unscaled = compute_model_metrics(y_test, pred_unscaled, prob_unscaled)

    experiments["Scaled_vs_Unscaled_Features"] = {
        "Scaled_Features": model_results["Logistic Regression"],
        "Unscaled_Features": metrics_unscaled
    }

    # Save artifacts
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model_pipeline.pkl"))
    
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(model_results, f, indent=2)

    with open(os.path.join(MODELS_DIR, "experiments_summary.json"), "w") as f:
        json.dump(experiments, f, indent=2)

    # Feature Importance / Key Words
    text_vectorizer = best_pipeline.named_steps['preprocessor'].named_transformers_['text']
    classifier_model = best_pipeline.named_steps['classifier']
    feature_indicators = extract_top_features(text_vectorizer, classifier_model)

    with open(os.path.join(MODELS_DIR, "feature_indicators.json"), "w") as f:
        json.dump(feature_indicators, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING FINISHED & ARTIFACTS SAVED TO models/")
    print("=" * 60)

    return model_results, experiments

if __name__ == "__main__":
    run_training_pipeline()
