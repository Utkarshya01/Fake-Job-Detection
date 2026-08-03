import os
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "fake_job_postings.csv")

# Public mirror raw URL for EMSCAD dataset if available
DATASET_URL = "https://raw.githubusercontent.com/datasets/employment-scam-dataset/main/fake_job_postings.csv"

def generate_synthetic_dataset(num_samples=2500, fraud_ratio=0.15):
    """
    Generates a realistic synthetic job postings dataset following exact Kaggle EMSCAD schema
    for training and testing when external download is unavailable or offline.
    """
    np.random.seed(42)
    n_fraud = int(num_samples * fraud_ratio)
    n_real = num_samples - n_fraud

    # Templates for Real Jobs
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

    # Templates for Fraudulent Jobs (Common scam characteristics: high pay for low effort, wire transfer, immediate hire, vague company profile, contact via personal email)
    fake_titles = [
        "Data Entry Clerk - Work From Home - Immediate Hire!", "Earn $5000/Week Online - No Experience Needed",
        "Executive Personal Assistant (Remote)", "Mystery Shopper / Financial Transfer Coordinator",
        "Package Dispatch Manager - High Pay", "Online Reviewer / Home Processor"
    ]
    fake_companies = [
        "",  # Often missing company profile
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
    employment_types = ["Full-time", "Part-time", "Contract", "Temporary", "Other"]
    experiences = ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Applicable"]
    educations = ["Bachelor's Degree", "High School or equivalent", "Master's Degree", "Unspecified"]

    records = []

    # Generate Real Jobs
    for i in range(n_real):
        has_logo = np.random.choice([1, 0], p=[0.85, 0.15])
        telecommute = np.random.choice([0, 1], p=[0.7, 0.3])
        has_q = np.random.choice([1, 0], p=[0.75, 0.25])
        sal = np.random.choice(["60000-90000", "80000-120000", "50000-75000", "100000-140000", ""], p=[0.3, 0.2, 0.2, 0.1, 0.2])

        records.append({
            "job_id": i + 1,
            "title": np.random.choice(real_titles),
            "location": f"US, {np.random.choice(['CA', 'NY', 'TX', 'WA', 'FL'])}, {np.random.choice(['San Francisco', 'New York', 'Austin', 'Seattle', 'Miami'])}",
            "department": np.random.choice(["Engineering", "Operations", "Sales", "Product", "Human Resources"]),
            "salary_range": sal,
            "company_profile": np.random.choice(real_companies),
            "description": np.random.choice(real_descriptions),
            "requirements": np.random.choice(real_requirements),
            "benefits": np.random.choice(real_benefits),
            "telecommuting": telecommute,
            "has_company_logo": has_logo,
            "has_questions": has_q,
            "employment_type": np.random.choice(employment_types, p=[0.6, 0.15, 0.15, 0.05, 0.05]),
            "required_experience": np.random.choice(experiences, p=[0.2, 0.3, 0.35, 0.1, 0.02, 0.03]),
            "required_education": np.random.choice(educations, p=[0.6, 0.2, 0.15, 0.05]),
            "industry": np.random.choice(industries),
            "function": np.random.choice(functions),
            "fraudulent": 0
        })

    # Generate Fraudulent Jobs
    for i in range(n_fraud):
        has_logo = np.random.choice([0, 1], p=[0.8, 0.2]) # Fake jobs rarely have logos
        telecommute = np.random.choice([1, 0], p=[0.75, 0.25]) # Frequently remote work-from-home scams
        has_q = np.random.choice([0, 1], p=[0.8, 0.2]) # Rarely have detailed screening questions
        sal = np.random.choice(["5000-10000", "100000-200000", "3000-5000", ""], p=[0.4, 0.2, 0.2, 0.2])

        records.append({
            "job_id": n_real + i + 1,
            "title": np.random.choice(fake_titles),
            "location": f"US, {np.random.choice(['CA', 'FL', 'OH', 'MI', 'NV'])}, Remote",
            "department": np.random.choice(["General", "Financial", "Home Office", "Data"]),
            "salary_range": sal,
            "company_profile": np.random.choice(fake_companies),
            "description": np.random.choice(fake_descriptions),
            "requirements": np.random.choice(fake_requirements),
            "benefits": np.random.choice(fake_benefits),
            "telecommuting": telecommute,
            "has_company_logo": has_logo,
            "has_questions": has_q,
            "employment_type": np.random.choice(employment_types, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
            "required_experience": np.random.choice(experiences, p=[0.5, 0.2, 0.1, 0.0, 0.0, 0.2]),
            "required_education": np.random.choice(educations, p=[0.2, 0.5, 0.05, 0.25]),
            "industry": np.random.choice(["Financial Services", "Customer Service", "Business Services", "Administrative"]),
            "function": np.random.choice(["Administrative", "Finance", "Customer Service"]),
            "fraudulent": 1
        })

    df = pd.DataFrame(records)
    # Shuffle dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_job_dataset():
    """Loads the job postings dataset. Attempts to load/download real EMSCAD data first, fallback to synthetic."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        print(f"[DatasetLoader] Loading dataset from existing file: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        return df

    # Attempt download
    try:
        print(f"[DatasetLoader] Downloading dataset from URL: {DATASET_URL}...")
        req = urllib.request.Request(DATASET_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response, open(DATA_FILE, 'wb') as out_file:
            out_file.write(response.read())
        df = pd.read_csv(DATA_FILE)
        print(f"[DatasetLoader] Download successful! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[DatasetLoader] Download failed or timed out ({e}). Generating high-quality synthetic EMSCAD benchmark dataset...")
        df = generate_synthetic_dataset(num_samples=3000, fraud_ratio=0.15)
        df.to_csv(DATA_FILE, index=False)
        print(f"[DatasetLoader] Synthetic dataset saved to {DATA_FILE}. Shape: {df.shape}")
        return df

if __name__ == "__main__":
    df = load_job_dataset()
    print("Class Distribution:")
    print(df['fraudulent'].value_counts(normalize=True))
