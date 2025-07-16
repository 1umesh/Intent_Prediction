You can visit live site at https://1umesh.github.io/intent_front/
 Backend API: [https://intent-back.onrender.com]
# 🧠 AI Lead Intent Score Dashboard

An end-to-end machine learning and rule-based LLM system that predicts the **intent score of leads** for brokers and sales teams. The project helps filter **high-intent** leads from low-quality ones, enabling faster conversions and better productivity.

## 🚀 Features

- 🎯 **Lead Intent Prediction** using a trained ML pipeline
- 🧠 **LLM-based Re-Ranker** to enhance scoring via rules
- 🌐 **FastAPI** backend serving predictions through API
- 💻 **Frontend Dashboard** for user-friendly interaction
- ✅ **Consent Checkbox** before submission (mock compliance)
- ☁️ **Deployed on**:
  - Frontend: GitHub Pages / Netlify
  - Backend: Render / Railway / Localhost

---

## 📁 Project Structure
<pre>
```
├── backend/
│   ├── main.py                  # FastAPI application
│   └── utils/
│       └── re_ranker.py         # Rule-based re-ranking logic
│
├── frontend/
│   ├── index.html               # User interface
│   ├── styles.css               # UI styling
│   └── script.js                # Frontend logic & API calls
│
├── train.py                     # Script to train ML model
├── customer_conversion_training_dataset.csv
├── conversion_pipeline.pkl      # Trained ML pipeline
├── requirements.txt             # Python dependencies
└── README.md
```
</pre>

---

## 🔍 How It Works

1. **User fills lead details** on the frontend form.
2. **Consent checkbox** must be checked to enable form submission.
3. Data is sent to **FastAPI backend**, where:
   - A **trained ML model** predicts initial intent score.
   - A **rule-based LLM re-ranker** adjusts the score based on defined business logic.
4. The final **intent score** is displayed on the dashboard.

---

## ⚙️ Tech Stack

- **Backend:** FastAPI, Joblib, Scikit-learn
- **Frontend:** HTML, CSS, JavaScript
- **Modeling:** Logistic Regression (or your chosen algorithm)
- **LLM:** Rule-based logic for re-ranking scores
- **Deployment:** GitHub Pages / Netlify (frontend), Render / Railway (backend)

---

## 🧪 How to Run Locally

### Backend

```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload
