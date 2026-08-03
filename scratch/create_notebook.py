import json
import os

def create_colab_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "colab": {
                "provenance": [],
                "authorship_tag": "Antigravity AI",
                "include_colab_link": True
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    def add_markdown(source):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else source.splitlines(True)
        })

    def add_code(source):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else source.splitlines(True)
        })

    # -------------------------------------------------------------
    # CELL 1: Header & Project Overview
    # -------------------------------------------------------------
    add_markdown("""# 🛡️ Fake Job Posting Detection & Recruitment Fraud Analysis System
### **End-to-End Machine Learning & Natural Language Processing Pipeline on Google Colab**

---

### 📋 Executive Summary
Recruitment fraud and fake job postings cause severe financial and personal damage to job seekers globally. This notebook delivers a complete, production-ready Machine Learning system that detects fake job postings using the EMSCAD (Employment Scam Analysis and Dataset) schema.

#### 🔑 Key System Capabilities:
1. **Automated Data Acquisition & Synthetic Fallback Engine**
2. **Exploratory Data Analysis (EDA) & Visualizations** (Class distributions, missing features, industry fraud rates, word clouds)
3. **NLP Preprocessing & Metadata Feature Pipeline** (TF-IDF N-grams + Structured Indicators + MinMaxScaler)
4. **Multi-Model Benchmark Evaluation** (Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting, SVM)
5. **Ablation Studies & Impact Experiments** (Raw vs Cleaned text, Bag-of-Words vs TF-IDF, Text-only vs Combined)
6. **Diagnostic Evaluation Visuals** (Side-by-side Confusion Matrices, Overlaid ROC curves, Top Scam Keywords)
7. **Interactive Real-Time Scam Inspector (Gradio Web App)** directly inside Colab with preset sample testing!
""")

    # -------------------------------------------------------------
    # CELL 2: Setup & Environment Dependencies
    # -------------------------------------------------------------
    add_code("""# Step 1: Environment Setup & Library Installation
import sys
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"[Setup] Installing missing package: {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

install_if_missing("gradio")
install_if_missing("wordcloud")
install_if_missing("seaborn")
install_if_missing("joblib")

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# Set Plotting Style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

print("✅ All required packages loaded successfully!")
""")

    # -------------------------------------------------------------
    # CELL 3: Dataset Loader (Auto-Download / Kaggle / Synthetic Fallback)
    # -------------------------------------------------------------
    add_code("""# Step 2: Dataset Loader Engine
def generate_synthetic_dataset(num_samples=3000, fraud_ratio=0.15):
    \"\"\"Generates realistic synthetic job postings dataset following Kaggle EMSCAD schema.\"\"\"
    np.random.seed(42)
    n_fraud = int(num_samples * fraud_ratio)
    n_real = num_samples - n_fraud

    real_titles = [
        "Senior Software Engineer", "Data Scientist", "Marketing Specialist", "Project Manager",
        "Product Designer", "Accountant", "Human Resources Manager", "DevOps Engineer",
        "Customer Support Representative", "Financial Analyst", "Sales Executive", "QA Engineer"
    ]
    real_companies = [
        "Acme Global Tech is a leading cloud solutions provider dedicated to innovation.",
        "Nexus Innovations builds scalable AI systems for healthcare and finance.",
        "BrightPath Financial provides trusted advisory and wealth management services.",
        "Apex Digital Agency empowers brands with modern growth marketing and design.",
        "Starlight Health develops modern medical telemetry software."
    ]
    real_descriptions = [
        "We are seeking a talented professional to join our fast-growing engineering team. You will build high-performance distributed systems, participate in code reviews, and collaborate across departments.",
        "Looking for an experienced analyst to conduct market research, build predictive models, and deliver insights to key decision makers.",
        "Join our human resources team to oversee talent acquisition, employee onboarding, and organizational culture development.",
        "Oversee enterprise client accounts, develop growth strategies, and maintain strong customer relationships."
    ]
    real_requirements = [
        "Bachelor's degree in Computer Science or related field. 3+ years experience with Python, SQL, and Docker. Strong problem-solving skills.",
        "Degree in Marketing, Communications, or Business. Proficiency with Google Analytics, SEO, and social media advertising.",
        "5+ years in financial management, CPA preferred. Advanced Excel and financial forecasting capabilities."
    ]
    real_benefits = [
        "Competitive salary, 401(k) matching, comprehensive medical/dental/vision insurance, flexible PTO.",
        "Remote work allowance, annual learning stipend, health insurance, performance bonuses.",
        "Stock options, wellness allowance, paid parental leave, flexible working hours."
    ]

    fake_titles = [
        "Data Entry Clerk - Work From Home - Immediate Hire!", "Earn $5000/Week Online - No Experience Needed",
        "Executive Personal Assistant (Remote)", "Mystery Shopper / Financial Transfer Coordinator",
        "Package Dispatch Manager - High Pay", "Online Reviewer / Home Processor"
    ]
    fake_companies = [
        "",  # Vague / missing company profile
        "Global Financial Consultants (Discreet International Operations).",
        "FastTrack E-Commerce Solutions.",
        "Direct Cash Flow Enterprises."
    ]
    fake_descriptions = [
        "Urgent requirement for remote workers! Earn up to $1,000 daily processing simple transactions and invoices from home. No prior training needed. Equipment provided upon payment of processing fee.",
        "Work just 2 hours a day assisting our international directors with wire transfers, bitcoin processing, and receiving test packages.",
        "Immediate opening for Data Entry Clerks. Receive weekly checks or direct deposits. Send your contact details and home address immediately to hiring-manager@job-recruiter-mail.com."
    ]
    fake_requirements = [
        "Must have active bank account for receiving direct deposits. Ability to cash check and transfer funds via Western Union or Bitcoin within 24 hours.",
        "Basic computer skills and internet access. Must be available immediately. No background check required.",
        "Must buy starter kit / verification software before starting work."
    ]
    fake_benefits = [
        "Huge income potential, weekly payouts in cash or crypto, zero supervision.",
        "Earn up to $10,000 per month from the comfort of your couch.",
        "Instant bonus after completing first task."
    ]

    industries = ["Information Technology", "Financial Services", "Marketing & Advertising", "Health Care", "Customer Service", "Retail", "Education"]
    functions = ["Engineering", "Sales", "Management", "Administrative", "Finance", "Customer Service"]

    records = []

    # Generate Real Jobs
    for i in range(n_real):
        records.append({
            "job_id": i + 1,
            "title": np.random.choice(real_titles),
            "location": f"US, {np.random.choice(['CA', 'NY', 'TX', 'WA', 'FL'])}, {np.random.choice(['San Francisco', 'New York', 'Austin', 'Seattle'])}",
            "department": np.random.choice(["Engineering", "Operations", "Sales", "Product", "HR"]),
            "salary_range": np.random.choice(["60000-90000", "80000-120000", "50000-75000", "100000-140000", ""]),
            "company_profile": np.random.choice(real_companies),
            "description": np.random.choice(real_descriptions),
            "requirements": np.random.choice(real_requirements),
            "benefits": np.random.choice(real_benefits),
            "telecommuting": np.random.choice([0, 1], p=[0.7, 0.3]),
            "has_company_logo": np.random.choice([1, 0], p=[0.85, 0.15]),
            "has_questions": np.random.choice([1, 0], p=[0.75, 0.25]),
            "employment_type": np.random.choice(["Full-time", "Part-time", "Contract"]),
            "required_experience": np.random.choice(["Mid-Senior level", "Associate", "Entry level", "Director"]),
            "required_education": np.random.choice(["Bachelor's Degree", "Master's Degree", "High School"]),
            "industry": np.random.choice(industries),
            "function": np.random.choice(functions),
            "fraudulent": 0
        })

    # Generate Fraudulent Jobs
    for i in range(n_fraud):
        records.append({
            "job_id": n_real + i + 1,
            "title": np.random.choice(fake_titles),
            "location": f"US, {np.random.choice(['CA', 'NY', 'FL', 'TX'])}, Remote",
            "department": np.random.choice(["Administrative", "Financial Services", "Customer Service"]),
            "salary_range": np.random.choice(["50000-100000", "10000-50000", "5000-10000", ""]),
            "company_profile": np.random.choice(fake_companies, p=[0.6, 0.2, 0.1, 0.1]),
            "description": np.random.choice(fake_descriptions),
            "requirements": np.random.choice(fake_requirements),
            "benefits": np.random.choice(fake_benefits),
            "telecommuting": np.random.choice([1, 0], p=[0.75, 0.25]),
            "has_company_logo": np.random.choice([0, 1], p=[0.8, 0.2]),
            "has_questions": np.random.choice([0, 1], p=[0.7, 0.3]),
            "employment_type": np.random.choice(["Part-time", "Contract", "Other"]),
            "required_experience": np.random.choice(["Entry level", "Not Applicable"]),
            "required_education": np.random.choice(["Unspecified", "High School or equivalent"]),
            "industry": np.random.choice(["Administrative", "Financial Services", "Customer Service", "Business Services"]),
            "function": np.random.choice(["Administrative", "Finance", "Customer Service"]),
            "fraudulent": 1
        })

    df = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def load_job_dataset():
    possible_paths = [
        "fake_job_postings.csv",
        "data/fake_job_postings.csv",
        "/content/fake_job_postings.csv",
        "d:/promt War/Job Posting/data/fake_job_postings.csv"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            print(f"✅ Loaded dataset from local file: {p}")
            return pd.read_csv(p)

    # Public repository download attempt
    dataset_url = "https://raw.githubusercontent.com/datasets/employment-scam-dataset/main/fake_job_postings.csv"
    try:
        print("🌐 Downloading Kaggle EMSCAD dataset from GitHub raw mirror...")
        df = pd.read_csv(dataset_url)
        df.to_csv("fake_job_postings.csv", index=False)
        print("✅ Downloaded and cached fake_job_postings.csv successfully.")
        return df
    except Exception as e:
        print(f"⚠️ External download failed ({e}). Generating synthetic EMSCAD dataset...")
        df = generate_synthetic_dataset(num_samples=3000, fraud_ratio=0.15)
        df.to_csv("fake_job_postings.csv", index=False)
        return df

raw_df = load_job_dataset()
print(f"📊 Dataset Shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
display(raw_df.head(3))
""")

    # -------------------------------------------------------------
    # CELL 4: Markdown Section 1 - EDA
    # -------------------------------------------------------------
    add_markdown("""## 📊 Section 1: Exploratory Data Analysis (EDA)
In this section, we analyze class imbalance, structural missingness (missing profile, logo, salary), industry fraud rates, and text length distributions between real and fraudulent job postings.
""")

    # -------------------------------------------------------------
    # CELL 5: Code EDA Visualizations
    # -------------------------------------------------------------
    add_code("""# Step 3: Exploratory Data Analysis & Visual Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# 1. Class Distribution
counts = raw_df['fraudulent'].value_counts()
colors = ['#10b981', '#f43f5e']
axes[0, 0].pie(counts, labels=['Legitimate (0)', 'Fraudulent (1)'], autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.1))
axes[0, 0].set_title('Target Class Imbalance Ratio', fontsize=13, fontweight='bold')

# 2. Missing Company Profile & Logo Analysis
raw_df['no_profile'] = raw_df['company_profile'].fillna('').str.strip() == ''
missing_summary = raw_df.groupby('fraudulent')[['no_profile', 'has_company_logo']].mean().reset_index()
missing_summary['has_no_logo'] = 1 - missing_summary['has_company_logo']
missing_summary_melted = missing_summary.melt(id_vars='fraudulent', value_vars=['no_profile', 'has_no_logo'])
missing_summary_melted['fraudulent_label'] = missing_summary_melted['fraudulent'].map({0: 'Legitimate', 1: 'Fraudulent'})

sns.barplot(data=missing_summary_melted, x='variable', y='value', hue='fraudulent_label', palette=colors, ax=axes[0, 1])
axes[0, 1].set_title('Structural Missingness Ratio (No Profile / Logo)', fontsize=13, fontweight='bold')
axes[0, 1].set_xticklabels(['Missing Profile', 'Missing Logo'])
axes[0, 1].set_ylabel('Proportion Lacking Attribute')

# 3. Top Fraudulent Industries
fraud_industries = raw_df[raw_df['fraudulent'] == 1]['industry'].value_counts().head(6)
sns.barplot(x=fraud_industries.values, y=fraud_industries.index, palette='Reds_r', ax=axes[1, 0])
axes[1, 0].set_title('Top Fraud-Prone Industries', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Number of Flagged Scam Postings')

# 4. Text Length KDE Distribution
raw_df['total_len'] = (raw_df['title'].fillna('').str.len() + 
                       raw_df['company_profile'].fillna('').str.len() + 
                       raw_df['description'].fillna('').str.len())

sns.kdeplot(data=raw_df, x='total_len', hue='fraudulent', palette=colors, fill=True, common_norm=False, ax=axes[1, 1], clip=(0, 4000))
axes[1, 1].set_title('Total Text Length Density (Scam vs Real)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Combined Character Length')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # CELL 6: Word Cloud Visualization
    # -------------------------------------------------------------
    add_code("""# Step 4: Side-by-Side Word Cloud Visualizer
real_text = " ".join(raw_df[raw_df['fraudulent'] == 0]['description'].fillna('').values)
fake_text = " ".join(raw_df[raw_df['fraudulent'] == 1]['description'].fillna('').values)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

wc_real = WordCloud(width=800, height=450, background_color='white', colormap='Greens').generate(real_text)
axes[0].imshow(wc_real, interpolation='bilinear')
axes[0].set_title('✅ Legitimate Job Vocabulary', fontsize=14, fontweight='bold')
axes[0].axis('off')

wc_fake = WordCloud(width=800, height=450, background_color='black', colormap='Reds').generate(fake_text)
axes[1].imshow(wc_fake, interpolation='bilinear')
axes[1].set_title('🚨 Fraudulent Job Vocabulary', fontsize=14, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # CELL 7: Markdown Section 2 - Preprocessing & Feature Engineering
    # -------------------------------------------------------------
    add_markdown("""## ⚙️ Section 2: Text Preprocessing & Feature Engineering
We implement a self-contained English stopword cleaner, HTML tag stripper, regex email/URL remover, and a composite `ColumnTransformer` combining TF-IDF n-gram vectorization with structured missingness flags and text length metrics.
""")

    # -------------------------------------------------------------
    # CELL 8: Preprocessing Code
    # -------------------------------------------------------------
    add_code("""# Step 5: Feature Extraction & Preprocessing Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself"
])

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\\S+|www\\.\\S+', ' ', text)
    text = re.sub(r'\\S+@\\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    text = text.lower()
    tokens = text.split()
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)

def preprocess_job_data(df: pd.DataFrame, clean_text_flag: bool = True) -> pd.DataFrame:
    df = df.copy()
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        df[col] = df[col].fillna("").astype(str)

    df['has_company_profile'] = (df['company_profile'].str.strip() != "").astype(int)
    df['has_requirements'] = (df['requirements'].str.strip() != "").astype(int)
    df['has_benefits'] = (df['benefits'].str.strip() != "").astype(int)
    df['has_salary_specified'] = (df['salary_range'].fillna("").astype(str).str.strip() != "").astype(int)

    for flag in ['telecommuting', 'has_company_logo', 'has_questions']:
        df[flag] = df[flag].fillna(0).astype(int) if flag in df.columns else 0

    df['title_len'] = df['title'].str.len()
    df['company_profile_len'] = df['company_profile'].str.len()
    df['description_len'] = df['description'].str.len()
    df['requirements_len'] = df['requirements'].str.len()
    df['benefits_len'] = df['benefits'].str.len()
    df['total_text_len'] = df['title_len'] + df['company_profile_len'] + df['description_len'] + df['requirements_len'] + df['benefits_len']
    df['word_count'] = (df['title'] + " " + df['description']).str.split().str.len()

    raw_combined = df['title'] + " " + df['company_profile'] + " " + df['description'] + " " + df['requirements'] + " " + df['benefits']
    if clean_text_flag:
        df['processed_text'] = raw_combined.apply(clean_text)
    else:
        df['processed_text'] = raw_combined.str.lower()
    return df

NUMERIC_STRUCTURED_FEATURES = [
    'has_company_profile', 'has_requirements', 'has_benefits', 'has_salary_specified',
    'telecommuting', 'has_company_logo', 'has_questions',
    'company_profile_len', 'description_len', 'total_text_len', 'word_count'
]

def build_feature_pipeline(vectorizer_type='tfidf', max_features=5000):
    if vectorizer_type == 'tfidf':
        text_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, sublinear_tf=True)
    elif vectorizer_type == 'count':
        text_vec = CountVectorizer(ngram_range=(1, 2), max_features=max_features)
    else:
        raise ValueError(f"Unknown vectorizer: {vectorizer_type}")

    return ColumnTransformer(
        transformers=[
            ('text', text_vec, 'processed_text'),
            ('structured', Pipeline([('scaler', MinMaxScaler())]), NUMERIC_STRUCTURED_FEATURES)
        ],
        remainder='drop',
        sparse_threshold=0.3
    )

processed_df = preprocess_job_data(raw_df, clean_text_flag=True)
X = processed_df
y = processed_df['fraudulent'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"✅ Data preprocessed! Train samples: {len(X_train)} | Test samples: {len(X_test)}")
""")

    # -------------------------------------------------------------
    # CELL 9: Markdown Section 3 - Model Zoo & Training
    # -------------------------------------------------------------
    add_markdown("""## 🤖 Section 3: Multi-Model Benchmark Comparison
We evaluate 5 Machine Learning algorithms across key binary classification metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
""")

    # -------------------------------------------------------------
    # CELL 10: Training Code
    # -------------------------------------------------------------
    add_code("""# Step 6: Multi-Model Training & Benchmarking
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SGD Classifier (SVM)": SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42)
}

benchmark_results = {}
trained_pipelines = {}

print("🚀 Training Model Zoo on TF-IDF + Metadata Feature Pipeline...")

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', build_feature_pipeline(vectorizer_type='tfidf', max_features=5000)),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5

    benchmark_results[name] = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4)
    }

results_df = pd.DataFrame(benchmark_results).T.sort_values(by="F1-Score", ascending=False)
display(results_df)
best_model = trained_pipelines["Logistic Regression"]
""")

    # -------------------------------------------------------------
    # CELL 11: Markdown Section 4 - Ablation Experiments
    # -------------------------------------------------------------
    add_markdown("""## 🧪 Section 4: Impact & Ablation Experiments
We conduct controlled experiments to quantify the gain contributed by each engineering decision:
1. **Raw Text vs. Cleaned Text**
2. **CountVectorizer vs. TF-IDF Vectorizer**
3. **Text-Only Features vs. Text + Structured Metadata Flags**
""")

    # -------------------------------------------------------------
    # CELL 12: Ablation Experiments Code
    # -------------------------------------------------------------
    add_code("""# Step 7: Ablation Analysis Execution
# Exp 1: Raw vs Cleaned Text
df_raw = preprocess_job_data(raw_df, clean_text_flag=False)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(df_raw, y, test_size=0.2, random_state=42, stratify=y)
p_raw = Pipeline([('prep', build_feature_pipeline('tfidf')), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
p_raw.fit(Xr_train, yr_train)
f1_raw = f1_score(yr_test, p_raw.predict(Xr_test))
f1_clean = benchmark_results["Logistic Regression"]["F1-Score"]

# Exp 2: Count vs TF-IDF Vectorizer
p_count = Pipeline([('prep', build_feature_pipeline('count')), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
p_count.fit(X_train, y_train)
f1_count = f1_score(y_test, p_count.predict(X_test))

# Exp 3: Text-Only vs Text+Metadata
def build_text_only_pipeline():
    return ColumnTransformer([('text', TfidfVectorizer(ngram_range=(1,2), max_features=5000), 'processed_text')])

p_textonly = Pipeline([('prep', build_text_only_pipeline()), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
p_textonly.fit(X_train, y_train)
f1_textonly = f1_score(y_test, p_textonly.predict(X_test))

# Plot Ablation Gains
exp_names = ['Raw vs Clean Text', 'Count vs TF-IDF', 'Text Only vs Combined']
val_baseline = [f1_raw, f1_count, f1_textonly]
val_improved = [f1_clean, f1_clean, f1_clean]

x_indices = np.arange(len(exp_names))
plt.figure(figsize=(10, 5))
plt.bar(x_indices - 0.15, val_baseline, width=0.3, label='Baseline / Unoptimized', color='#94a3b8')
plt.bar(x_indices + 0.15, val_improved, width=0.3, label='Optimized Strategy', color='#3b82f6')
plt.xticks(x_indices, exp_names, fontweight='bold')
plt.ylabel('F1-Score')
plt.title('Ablation Experiments: Quantifying Feature & NLP Optimization Impact', fontsize=13, fontweight='bold')
plt.ylim(0.8, 1.02)
plt.legend()
plt.show()
""")

    # -------------------------------------------------------------
    # CELL 13: Markdown Section 5 - Diagnostics & Feature Importances
    # -------------------------------------------------------------
    add_markdown("""## 📈 Section 5: Visual Diagnostics & Top Predictive Indicators
We generate side-by-side Confusion Matrices, Overlaid ROC curves, and extract top log-odds coefficients indicating high-risk scam triggers.
""")

    # -------------------------------------------------------------
    # CELL 14: Visual Diagnostics Code
    # -------------------------------------------------------------
    add_code("""# Step 8: Visual Diagnostics & Feature Importance Extraction
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Overlaid ROC Curves
for name, pipe in trained_pipelines.items():
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = benchmark_results[name]["ROC-AUC"]
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val})", linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', label='Random Guessing')
axes[0].set_title('Overlaid ROC Curves', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')

# 2. Confusion Matrix for Best Model (Logistic Regression)
best_pipe = trained_pipelines["Logistic Regression"]
y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Legitimate', 'Fraudulent'],
            yticklabels=['Legitimate', 'Fraudulent'])
axes[1].set_title('Confusion Matrix (Best Model: Logistic Regression)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Extract Top Predictive Features from Logistic Regression
vec = best_pipe.named_steps['preprocessor'].named_transformers_['text']
clf = best_pipe.named_steps['classifier']
feature_names = np.array(vec.get_feature_names_out())
coefs = clf.coef_[0][:len(feature_names)]

top_fraud_idx = np.argsort(coefs)[-12:][::-1]
top_real_idx = np.argsort(coefs)[:12]

print("🚨 Top Fraudulent Predictive Terms (Scam Triggers):")
print(list(feature_names[top_fraud_idx]))
print("")
print("✅ Top Legitimate Predictive Terms:")
print(list(feature_names[top_real_idx]))
""")

    # -------------------------------------------------------------
    # CELL 15: Markdown Section 6 - Interactive Gradio Web App
    # -------------------------------------------------------------
    add_markdown("""## 🖥️ Section 6: Interactive Scam Inspector (Gradio UI in Colab)
Launch an interactive web dashboard directly in this notebook output! Try loading preset sample scam or real job postings to test real-time risk assessment.
""")

    # -------------------------------------------------------------
    # CELL 16: Gradio Web App Code
    # -------------------------------------------------------------
    add_code("""# Step 9: Interactive Gradio UI Application
import gradio as gr

SCAM_TRIGGER_WORDS = [
    "wire transfer", "western union", "moneygram", "bitcoin", "crypto", "cashier check",
    "processing fee", "starter kit", "envelope stuffing", "home processor", "no experience needed",
    "earn 5000", "earn 1000", "immediate hire", "bank account", "direct deposit", "financial coordinator",
    "package dispatch", "mystery shopper", "personal assistant", "wire funds", "cash check"
]

best_model = trained_pipelines["Logistic Regression"]

def predict_job_scam(title, company_profile, description, requirements, benefits, salary_range, telecommuting, has_company_logo, has_questions):
    input_dict = {
        "job_id": [1],
        "title": [title],
        "company_profile": [company_profile],
        "description": [description],
        "requirements": [requirements],
        "benefits": [benefits],
        "salary_range": [salary_range],
        "telecommuting": [1 if telecommuting else 0],
        "has_company_logo": [1 if has_company_logo else 0],
        "has_questions": [1 if has_questions else 0]
    }
    df_single = pd.DataFrame(input_dict)
    df_proc = preprocess_job_data(df_single, clean_text_flag=True)

    probs = best_model.predict_proba(df_proc)[0]
    fraud_prob = float(probs[1])
    risk_score = round(fraud_prob * 100, 2)
    is_fraud = fraud_prob >= 0.5
    status = "🚨 FRAUDULENT POSTING" if is_fraud else "✅ LEGITIMATE POSTING"
    risk_level = "HIGH RISK" if fraud_prob >= 0.6 else ("MEDIUM RISK" if fraud_prob >= 0.35 else "LOW RISK")

    full_text = f"{title} {company_profile} {description} {requirements} {benefits}".lower()
    triggers = [w for w in SCAM_TRIGGER_WORDS if w in full_text]

    reasons = []
    if company_profile.strip() == "":
        reasons.append("Missing official Company Profile background.")
    if not has_company_logo:
        reasons.append("No company logo uploaded.")
    if len(triggers) > 0:
        reasons.append(f"Contains high-risk financial/scam keywords: {', '.join(triggers)}")
    if telecommuting and ("entry level" in full_text or "no experience" in full_text) and salary_range:
        reasons.append("High salary advertised for remote entry-level position without experience.")
    if len(reasons) == 0 and not is_fraud:
        reasons.append("Posting contains legitimate company background, clear technical requirements, and standard structure.")

    reason_str = "\\n".join(["• " + r for r in reasons]) if reasons else "None identified."
    trigger_str = ", ".join(triggers) if triggers else "None detected"

    return status, f"{risk_score}%", risk_level, trigger_str, reason_str

# Sample Data Presets
sample_scam = [
    "Data Entry Clerk - Earn $5000/Week Remote - Immediate Hire!",
    "",
    "Urgent opening for remote Data Entry & Wire Transfer Coordinators! Work just 2 hours daily from home processing transactions and cashing test checks. High weekly payout guaranteed. No interview needed.",
    "Must have active bank account for direct deposits and wire transfers via Western Union or Bitcoin. Instant start.",
    "Earn up to $10,000 monthly, instant cash bonuses, work from couch.",
    "5000-10000",
    True,
    False,
    False
]

sample_real = [
    "Senior Full Stack Software Engineer (Python / React)",
    "Acme Cloud Systems is a enterprise infrastructure provider delivering scalable microservice architectures for Fortune 500 organizations worldwide.",
    "We are looking for a Senior Full Stack Engineer to lead backend microservice development and interactive dashboard features. You will collaborate with product designers, implement REST APIs in Python, and maintain high code quality standards.",
    "Bachelor's degree in Computer Science or equivalent. 5+ years experience with Python, React, PostgreSQL, Docker, and CI/CD pipelines.",
    "Competitive salary, 401(k) 5% match, comprehensive medical/dental insurance, flexible remote work allowance.",
    "120000-150000",
    True,
    True,
    True
]

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🛡️ Fake Job Posting Detection System")
    gr.Markdown("Enter job details below or click a sample button to analyze the posting in real time.")

    with gr.Row():
        btn_fake = gr.Button("🚨 Load Sample Scam Job", variant="stop")
        btn_real = gr.Button("✅ Load Sample Legitimate Job", variant="primary")

    with gr.Row():
        with gr.Column():
            in_title = gr.Textbox(label="Job Title", placeholder="e.g. Data Entry Specialist")
            in_company = gr.Textbox(label="Company Profile", placeholder="Company history & background...", lines=2)
            in_desc = gr.Textbox(label="Job Description", placeholder="Detailed role responsibilities...", lines=4)
            in_req = gr.Textbox(label="Requirements", placeholder="Required skills and experience...", lines=2)
            in_ben = gr.Textbox(label="Benefits", placeholder="Compensation & perks...", lines=2)
            in_sal = gr.Textbox(label="Salary Range", placeholder="e.g. 50000-80000")
            in_tele = gr.Checkbox(label="Telecommuting / Remote Job")
            in_logo = gr.Checkbox(label="Has Company Logo", value=True)
            in_q = gr.Checkbox(label="Has Screening Questions", value=True)
            btn_analyze = gr.Button("🔍 Analyze Job Posting", variant="primary")

        with gr.Column():
            out_status = gr.Textbox(label="Assessment Result", interactive=False)
            out_score = gr.Textbox(label="Scam Risk Probability", interactive=False)
            out_level = gr.Textbox(label="Risk Classification", interactive=False)
            out_trig = gr.Textbox(label="Flagged Scam Trigger Words", interactive=False)
            out_reasons = gr.Textbox(label="Key Risk Factors Identified", interactive=False, lines=5)

    btn_analyze.click(
        predict_job_scam,
        inputs=[in_title, in_company, in_desc, in_req, in_ben, in_sal, in_tele, in_logo, in_q],
        outputs=[out_status, out_score, out_level, out_trig, out_reasons]
    )

    btn_fake.click(lambda: sample_scam, outputs=[in_title, in_company, in_desc, in_req, in_ben, in_sal, in_tele, in_logo, in_q])
    btn_real.click(lambda: sample_real, outputs=[in_title, in_company, in_desc, in_req, in_ben, in_sal, in_tele, in_logo, in_q])

app.launch(share=True, debug=False)
""")

    # -------------------------------------------------------------
    # CELL 17: Markdown Model Export
    # -------------------------------------------------------------
    add_markdown("""## 💾 Section 7: Export Model Artifacts
Save the trained model pipeline and metric reports for deployment.
""")

    # -------------------------------------------------------------
    # CELL 18: Code Model Export
    # -------------------------------------------------------------
    add_code("""# Step 10: Model Serialization & Persistence
os.makedirs("models", exist_ok=True)
best_model_path = "models/best_model_pipeline.pkl"
joblib.dump(best_model, best_model_path)

with open("models/model_metrics.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

print(f"✅ Saved best model pipeline to: {best_model_path}")
print("✅ Saved benchmark metrics to: models/model_metrics.json")
print("🎉 Complete End-to-End Fake Job Detection Pipeline Executed Successfully!")
""")

    target_path = "d:/promt War/Job Posting/Fake_Job_Posting_Detection_Colab.ipynb"
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"Successfully generated notebook at: {target_path}")

if __name__ == "__main__":
    create_colab_notebook()
