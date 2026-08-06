# 🛡️ Fake Job Posting Detection using Machine Learning

> An AI-powered system that identifies fraudulent job postings using Natural Language Processing (NLP) and Machine Learning techniques.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-Interactive-red?style=for-the-badge)

---

## 📌 Project Overview

Fake job advertisements have become increasingly common, leading to financial loss and identity theft for job seekers. This project leverages **Artificial Intelligence, Machine Learning, and Natural Language Processing (NLP)** to automatically distinguish between legitimate and fraudulent job postings.

The system preprocesses textual job data, extracts meaningful features using **TF-IDF**, trains multiple machine learning models, and predicts whether a job posting is **Real** or **Fake**.

---

## 🎯 Objectives

- Detect fraudulent job postings accurately.
- Reduce manual verification efforts.
- Compare the performance of multiple machine learning algorithms.
- Build an interactive prediction interface for real-time testing.

---

# 🚀 Features

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Preprocessing
- 📝 Text Processing using NLP
- 🔤 TF-IDF Feature Extraction
- 🤖 Multiple Machine Learning Models
- 📈 Model Performance Comparison
- 📉 ROC Curve & Performance Metrics
- ⭐ Feature Importance Analysis
- 🌐 Interactive Gradio Interface
- 💾 Model Saving for Future Predictions

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Gradio
- Joblib

---

# 📂 Project Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Text Preprocessing
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Model Training
   │
   ▼
Performance Evaluation
   │
   ▼
Prediction Interface
```

---

# 🤖 Machine Learning Models

The project compares multiple machine learning algorithms including:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier
- Gradient Boosting Classifier
- SGD Classifier

Each model is evaluated using various performance metrics to identify the best-performing classifier.

---

# 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

# 📁 Project Structure

```
Fake-Job-Posting-Detection/
│
├── Fake_Job_Posting_Detection_Colab.ipynb
├── dataset/
├── saved_models/
├── images/
├── requirements.txt
└── README.md
```

---

# ▶️ Getting Started

### Clone the Repository

```bash
git clone https://github.com/yourusername/Fake-Job-Posting-Detection.git
```

### Navigate to the Project

```bash
cd Fake-Job-Posting-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Notebook

Open:

```
Fake_Job_Posting_Detection_Colab.ipynb
```

using

- Google Colab
- Jupyter Notebook

---

# 💡 Future Improvements

- Deploy using Streamlit or Flask
- Integrate BERT/Transformer models
- Explain predictions using SHAP/LIME
- Real-time API integration
- Improve model accuracy through hyperparameter tuning

---

# 📸 Screenshots

> Add screenshots of:
- EDA Visualizations
- Confusion Matrix
- ROC Curve
- Gradio Interface

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Utkarsh Yadav**

If you found this project useful, don't forget to ⭐ the repository!

---

## ⭐ Support

If you like this project:

🌟 Star this repository

🍴 Fork it

📢 Share it with others

Happy Coding! 🚀
