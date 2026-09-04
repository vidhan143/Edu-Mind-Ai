# 🎓 AI Student Intelligence & Academic Risk Analytics Platform

An advanced Machine Learning and AI-based student analytics system built using Streamlit, Scikit-Learn, and Python.

The platform predicts student performance, analyzes academic risks, performs clustering, and provides AI-driven recommendations using multiple ML algorithms.

## 🔗 Web Application Link - https://edumind-study-ai.streamlit.app

---

# 🚀 Features

## 📊 Dashboard Analytics
- Student performance overview
- Academic statistics
- Interactive charts
- Risk indicators

## 📈 Machine Learning Models
- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression
- Naive Bayes
- K-Means Clustering
- PCA Visualization

## 🤖 AI Features
- Student risk prediction
- Feedback sentiment analysis
- Performance categorization
- AI recommendation engine

## 🎨 Modern UI
- Dark theme dashboard
- Interactive visualizations
- Professional Streamlit interface
- Responsive layout

---

# 🛠 Technologies Used

- Python 3.12
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Plotly
- Matplotlib
- Seaborn
- NLTK

---

# 📂 Project Structure

```bash
EduMind-AI/
│
├── .devcontainer/               
│   └── devcontainer.json        # Development container configuration
│
├── .env                         # Stores secret API keys locally
│
├── .gitignore                   # Prevents unnecessary/private files from uploading
│
├── README.md                    # Main project documentation
│
├── APPLICATION OVERVIEW.md      # Detailed explanation of modules/features
│
├── requirements.txt             # Python dependencies for deployment
│
├── runtime.txt                  # Forces Streamlit to use Python 3.12
│
├── app/                         
│   │
│   └── app.py                   # Main Streamlit dashboard application
│
├── data/                        
│   │
│   ├── StudentsPerformance.csv  # Academic performance dataset
│   │
│   └── Student Mental health.csv
│       # Mental health and student wellness dataset
│
├── notebooks/                   
│   │
│   └── AI_Student_Analytics.ipynb
│       # Jupyter notebook for ML experimentation and analysis
│
└── __pycache__/                 # Auto-generated Python cache (ignore/delete)

```
# ⚙️ Complete Setup Guide

## 1️⃣ Go To Project Folder

## 2️⃣ Check Python Version

```bash
python3 --version
```

## 3️⃣ Install Python 3.12

```bash
brew install python@3.12
```

## 4️⃣ Verify Python 3.12

```bash
python3.12 --version
```

### Expected Output

```bash
Python 3.12.x
```

## 5️⃣ Create Virtual Environment

```bash
python3.12 -m venv venv
```

## 6️⃣ Activate Virtual Environment

```bash
source venv/bin/activate
```

## 7️⃣ Verify Python Inside Venv

```bash
python --version
```

### Expected Output

```bash
Python 3.12.x
```

## 8️⃣ Install Requirements

```bash
python -m pip install -r app/requirements.txt
```

## 9️⃣ Upgrade pip

```bash
pip install --upgrade pip
```

# ▶️ Run Streamlit Application

```bash
python -m streamlit run app/app.py
```

# 🌐 Streamlit URLs

After running the app:

```bash
Local URL: http://localhost:8501
Network URL: http://172.xx.xx.xx:8501
```
