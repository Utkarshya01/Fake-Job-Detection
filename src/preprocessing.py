import re
import pandas as pd
import numpy as np

# Standard English Stopwords list (self-contained, no external NLTK download dependency required)
STOPWORDS = set([
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could',
    'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s',
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m',
    'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t',
    'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t',
    'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too',
    'under', 'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t',
    'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s',
    'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself',
    'yourselves'
])

def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Cleans raw text by removing HTML, URLs, emails, special chars, and optionally stopwords."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # Remove non-alphabetical characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Normalize whitespaces
    tokens = text.split()

    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 2]

    return " ".join(tokens)


def preprocess_job_data(df: pd.DataFrame, clean_text_flag: bool = True) -> pd.DataFrame:
    """
    Performs full preprocessing on the job dataset:
    - Handles missing textual fields
    - Extracts structured binary & length features
    - Combines text attributes into a unified text corpus
    - Cleans combined text if clean_text_flag is True
    """
    df = df.copy()

    # Ensure essential columns exist
    text_cols = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    # 1. Feature Extraction: Missingness & Structured Indicators
    df['has_company_profile'] = (df['company_profile'].str.strip() != "").astype(int)
    df['has_requirements'] = (df['requirements'].str.strip() != "").astype(int)
    df['has_benefits'] = (df['benefits'].str.strip() != "").astype(int)
    
    if 'salary_range' in df.columns:
        df['has_salary_specified'] = (df['salary_range'].fillna("").astype(str).str.strip() != "").astype(int)
    else:
        df['has_salary_specified'] = 0

    # Pass-through binary flags
    for flag_col in ['telecommuting', 'has_company_logo', 'has_questions']:
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].fillna(0).astype(int)
        else:
            df[flag_col] = 0

    # Text length & word count features
    df['title_len'] = df['title'].str.len()
    df['company_profile_len'] = df['company_profile'].str.len()
    df['description_len'] = df['description'].str.len()
    df['requirements_len'] = df['requirements'].str.len()
    df['benefits_len'] = df['benefits'].str.len()
    
    df['total_text_len'] = (df['title_len'] + df['company_profile_len'] + 
                            df['description_len'] + df['requirements_len'] + 
                            df['benefits_len'])

    # 2. Combine textual attributes into single unified corpus
    df['raw_combined_text'] = (
        df['title'] + " " +
        df['company_profile'] + " " +
        df['description'] + " " +
        df['requirements'] + " " +
        df['benefits']
    )

    # Clean text if requested
    if clean_text_flag:
        df['processed_text'] = df['raw_combined_text'].apply(lambda x: clean_text(x, remove_stopwords=True))
    else:
        df['processed_text'] = df['raw_combined_text'].apply(lambda x: clean_text(x, remove_stopwords=False))

    df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))

    return df

if __name__ == "__main__":
    from dataset_loader import load_job_dataset
    df = load_job_dataset()
    processed_df = preprocess_job_data(df)
    print("Preprocessing Sample:")
    print(processed_df[['has_company_profile', 'has_salary_specified', 'total_text_len', 'word_count']].head())
