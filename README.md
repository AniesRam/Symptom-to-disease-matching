# AI-Driven Symptom-to-Disease Matching & Hospital Management System

A full-stack web-based healthcare platform designed to optimize appointment scheduling, streamline clinical triage, and manage emergency internal alerts. Developed as a Final Year Project (FYP) for the Bachelor of Computer Science with Honours (Data Science) at Universiti Malaysia Sabah (UMS)[cite: 4, 8].

> **Disclaimer:** Developed for academic research and clinical decision support purposes only. It is intended to assist medical professionals with pre-consultation preparation and does not replace official clinical diagnosis[cite: 6, 20].

---

## 🚀 Key Features & Modules
* **AI Symptom Checker:** Analyzes patient-reported symptoms using a supervised machine learning model to predict prospective respiratory conditions with confidence scores[cite: 8, 41].
* **Automated Appointment Scheduling & Queue Management:** Matches patient booking requests with active doctor shift schedules and tracks real-time queue positions[cite: 24, 41].
* **Role-Based Access Control (RBAC):** Distinct secure portals and dashboards tailored for **Patients**, **Doctors/Nurses**, and **Administrators**[cite: 7, 21, 55].
* **Staff-Triggered Emergency Alert Module:** Allows authorized medical personnel to trigger internal alerts for high-risk or infectious cases, notifying staff to initiate rapid containment and sanitization protocols[cite: 8, 41].
* **Analytics & Reporting:** Integrated dashboard reporting tracking patient volume, department workloads, and common symptom trends[cite: 8, 22].

---

## 🛠️ Tech Stack & Architecture
* **Frontend:** HTML5, CSS3, JavaScript, Jinja2 Templates (Flask UI)[cite: 54]
* **Backend:** Python, Flask framework[cite: 8, 79]
* **Database & ORM:** PostgreSQL managed via SQLAlchemy ORM and pgAdmin 4[cite: 79]
* **Machine Learning:** Scikit-learn (Random Forest & Logistic Regression), pandas, NumPy, joblib[cite: 79]
* **Analytics Reporting:** Microsoft Power BI (`.pbix` report included)[cite: 8, 79]

---

## 🤖 Machine Learning Performance & Validation
To build an efficient triage model capable of running smoothly on standard 8GB RAM laptops, the data scope was fine-tuned around key respiratory conditions using the Kaggle *Diseases and Symptoms Dataset*[cite: 8, 23, 68].

* **Model Comparison:** Evaluated against a baseline Logistic Regression model (82.73% accuracy)[cite: 84].
* **Selected Model:** **Random Forest Classifier** was chosen for production due to superior handling of multi-class, overlapping symptom patterns[cite: 69].
* **Benchmark Accuracy:** Achieved **96.09% accuracy** and a weighted F1-score of **0.9605**[cite: 8, 70].
* **Artifacts:** Pre-trained model weights (`disease_model.pkl`) and feature configurations (`symptom_columns.pkl`) are serialized and loaded directly on application startup[cite: 36].

---

## 📊 System Evaluation
The system prototype underwent structured usability evaluation:
* **System Usability Scale (SUS):** Evaluated across 20 diverse respondents, achieving an overall score of **74.5** (classified within the "Good" usability range)[cite: 81, 103].
* **Stakeholder Review:** Validated through qualitative feedback interviews with healthcare management representatives from Pusat Rawatan Warga (PRW UMS)[cite: 90].

---

## 📁 Repository Structure
```text
├── datasets/            # Processed CSV training datasets
├── notebooks/           # Jupyter notebooks for model experimentation (Random Forest vs Logistic Regression)
├── templates/           # Jinja2 HTML frontend views for all user roles
├── app.py               # Main Flask application and route controllers
├── disease_model.pkl    # Serialized Random Forest model weights
└── symptom_columns.pkl  # Encoded feature mapping schema

add note: i also use PostgreSQL for this. because the database auto update so its easy for me :)
