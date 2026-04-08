# Demo:

https://github.com/user-attachments/assets/2f970a85-7ad4-40f0-9f13-4df257a0d19f


# 🎯 AI-Powered Student Career Readiness & Skill Gap Analysis System

## 📌 Project Overview

This project is an AI-powered web application designed to help students evaluate their **career readiness**, identify **skill gaps**, and prepare for **technical interviews**. It combines **Machine Learning** with **AI-based mock interviews** to provide a real-world career guidance system.

---

## 🌍 Problem Statement

Many students:

* Are unsure if they are industry-ready
* Lack awareness of required skills
* Apply for jobs without preparation
* Face rejection due to skill gaps

This system solves these problems by analyzing student profiles and providing actionable insights.

---

## 🚀 Features

* ✅ Career readiness prediction using Machine Learning
* ✅ Skill gap analysis based on company requirements
* ✅ Skill Match Score calculation
* ✅ Personalized learning recommendations
* ✅ AI-based Mock Interview generation
* ✅ Clean and simple web interface using Flask

---

## 🧠 Machine Learning Model

* **Type:** Binary Classification
* **Algorithms Used:** Random Forest / Logistic Regression
* **Output:**

  * 🟢 Career Ready
  * 🔴 Not Ready

---

## 📊 Input Features

* Academic Year
* CGPA
* Programming Languages
* Technologies Known
* Technical Skill Rating (1–5)
* Soft Skill Rating (1–5)
* Number of Projects
* Internship Experience (Yes/No)
* Weekly Upskilling Hours
* Career Interest
* Target Company

---

## 📈 Skill Match Score

Skill Match Score is calculated as:

```
Skill Match Score = (Matched Skills / Required Skills) × 100
```

This helps in identifying how well a student matches company requirements.

---

## 🤖 AI Mock Interview

The system uses an AI API to generate:

* Technical Questions
* Coding Questions
* HR Questions
* Sample Answers
* Feedback for improvement

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### Machine Learning

* Pandas
* Scikit-learn
* Joblib

### Frontend

* HTML
* CSS
* JavaScript

### AI Integration

* Gemini API / OpenAI API

---

## 📁 Project Structure

```
student-career-ai-system/
│
├── app.py
├── train_model.py
├── model.pkl
├── dataset.csv
├── requirements.txt
├── config.py
│
├── templates/
│   ├── index.html
│   ├── result.html
│   └── interview.html
│
├── static/
│   ├── style.css
│   └── script.js
│
├── utils/
│   ├── skill_match.py
│   └── recommendation.py
│
└── api/
    └── interview_api.py
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/student-career-ai-system.git
cd student-career-ai-system
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Add API Key

Create a `config.py` file:

```
API_KEY = "your_api_key_here"
```

---

### 5️⃣ Run the Application

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## 📊 Example Output

```
Student Name: Lavanya
Career Interest: Data Scientist
Target Company: Google

Skill Match Score: 60%

Status: Not Ready

Missing Skills:
- Data Structures
- System Design

Recommendations:
- Practice DSA daily
- Build ML projects
- Improve SQL skills
```

---

## 🎯 Future Enhancements

* Resume Analyzer
* Company Recommendation System
* Interview Scoring System
* Dashboard with graphs
* User login system

---

## 📄 Resume Description

Developed an AI-powered Student Career Readiness Prediction System using Machine Learning and Flask that analyzes skill gaps, recommends learning paths, and conducts AI-based mock interviews using API integration.

---

## 👩‍💻 Author

Lavanya Palla
B.Tech CSE (AI) Student

---

## ⭐ Conclusion

This project demonstrates how AI and Machine Learning can be used to solve real-world student career challenges by providing intelligent insights and interview preparation support.
