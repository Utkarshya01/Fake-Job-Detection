import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_STRUCTURED_FEATURES = [
    'has_company_profile', 'has_requirements', 'has_benefits', 'has_salary_specified',
    'telecommuting', 'has_company_logo', 'has_questions',
    'company_profile_len', 'description_len', 'total_text_len', 'word_count'
]

def build_feature_pipeline(
    vectorizer_type: str = 'tfidf',
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    use_scaling: bool = True
) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer pipeline combining unstructured text vectorization
    (TF-IDF or CountVectorizer) with scaled structured metadata features.
    """
    if vectorizer_type == 'tfidf':
        text_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True
        )
    elif vectorizer_type == 'count':
        text_vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features
        )
    else:
        raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}")

    if use_scaling:
        structured_pipeline = Pipeline([
            ('scaler', MinMaxScaler())
        ])
    else:
        structured_pipeline = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_vectorizer, 'processed_text'),
            ('structured', structured_pipeline, NUMERIC_STRUCTURED_FEATURES)
        ],
        remainder='drop',
        sparse_threshold=0.3
    )

    return preprocessor

if __name__ == "__main__":
    from dataset_loader import load_job_dataset
    from preprocessing import preprocess_job_data

    df = load_job_dataset()
    df_prep = preprocess_job_data(df)

    pipeline = build_feature_pipeline(vectorizer_type='tfidf', max_features=1000)
    X_feat = pipeline.fit_transform(df_prep)
    print(f"Feature matrix shape: {X_feat.shape}")
