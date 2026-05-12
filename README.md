# 🧠 Employee Turnover Prediction

A machine learning classification project that predicts whether an employee is likely to leave an organization, using **Logistic Regression** on real HR data. Built end-to-end in Python with feature engineering, model evaluation, and a comparative analysis with ElasticNet.

---

## 📌 Problem Statement

Employee attrition is one of the most costly challenges organizations face. Replacing a single employee can cost **50–200% of their annual salary** when factoring in recruitment, onboarding, and lost productivity.

This project uses supervised machine learning to identify employees at risk of leaving — giving HR teams the signal they need to act proactively, before it's too late.

---

## 📂 Project Structure

```
employee-turnover-prediction/
│
├── Logiestic_regresiin.ipynb     # Main notebook — full pipeline
├── employee_turnover.csv         # Dataset
└── README.md                     # Project documentation
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `Job_Satisfaction` | Employee's job satisfaction score |
| `Performance_Rating` | Performance rating given by manager |
| `Years_At_Company` | Number of years spent at the company |
| `Work_Life_Balance` | Work-life balance rating |
| `Distance_From_Home` | Commute distance from home |
| `Monthly_Income` | Monthly salary of the employee |
| `Education_Level` | Highest education level attained |
| `Age` | Employee age |
| `Num_Companies_Worked` | Number of previous companies worked at |
| `Employee_Role` | Job role/designation |
| `Annual_Bonus` | Annual bonus received |
| `Training_Hours` | Hours spent in training |
| `Department` | Department the employee belongs to |
| `Annual_Bonus_Squared` | Engineered feature — captures non-linear bonus effect |
| `Annual_Bonus_Training_Hours_Interaction` | Engineered feature — interaction between bonus and training |
| `Employee_Turnover` | **Target variable** — 1 (left) or 0 (stayed) |

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn
- **Environment:** Jupyter Notebook

---

## 🔁 Project Pipeline

```
1. Import Libraries
       ↓
2. Load Dataset (employee_turnover.csv)
       ↓
3. Exploratory Data Analysis
   └── Seaborn lineplot: Employee Turnover vs Job Satisfaction
       ↓
4. Feature Engineering
   └── Annual_Bonus_Squared
   └── Annual_Bonus_Training_Hours_Interaction
       ↓
5. Train-Test Split (80/20, random_state=42)
       ↓
6. Model Training
   └── Logistic Regression (max_iter=1500)
   └── ElasticNet (alpha sweep: 0.001 → 40)
       ↓
7. Evaluation
   └── Accuracy, Precision, MSE
```

---

## 📈 Model Results

### ✅ Logistic Regression

| Metric | Score |
|---|---|
| Accuracy | **85.93%** |
| Precision | **87.17%** |
| MSE | 0.1407 |

> Training samples: 533 &nbsp;|&nbsp; Test samples: 270 &nbsp;|&nbsp; Split: 80/20

---

### 🔁 ElasticNet — Alpha Sweep

| Alpha | MSE |
|---|---|
| 0.001 | 0.1117 |
| 0.1 | 0.2465 |
| 1 | 0.2505 |
| 2–40 | ~0.2505 (plateau) |

---

## 🏆 Model Recommendation

**Logistic Regression is the correct and recommended model for this problem.**

Employee turnover is a **binary classification task** (stay = 0, leave = 1). ElasticNet is a linear *regression* model designed for continuous outputs — applying it to binary labels produces misleading MSE values that are not a valid performance measure for this task.

| | Logistic Regression | ElasticNet |
|---|---|---|
| Problem type fit | ✅ Binary classification | ❌ Regression model |
| Output type | Class labels (0/1) | Continuous values |
| Accuracy | 85.93% | N/A |
| Precision | 87.17% | N/A |
| Recommended | ✅ Yes | ❌ No |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/RoHaN-0053/employee-turnover-prediction.git
cd employee-turnover-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**3. Launch the notebook**
```bash
jupyter notebook Logiestic_regresiin.ipynb
```

---

## 💡 Key Insights

- Employees with **lower job satisfaction** are significantly more likely to leave
- The Seaborn lineplot revealed a clear **positive relationship** between turnover and job satisfaction score among those who left
- **Feature engineering** (bonus squared + interaction terms) helped capture non-linear patterns the raw features alone would have missed
- When it predicted an employee would leave, the model was **right ~9 out of 10 times** (87.17% precision)

---

## 🔮 Future Improvements

- [ ] Add confusion matrix, recall, F1-score, and ROC-AUC evaluation
- [ ] Handle class imbalance using SMOTE or class weights
- [ ] Encode categorical features (`Department`, `Employee_Role`) properly using `pd.get_dummies()`
- [ ] Apply `StandardScaler` before training for better Logistic Regression performance
- [ ] Compare against proper classifiers — Random Forest, SVM, XGBoost
- [ ] Add markdown cells to notebook for better readability
- [ ] Build a simple Streamlit app for HR teams to input employee data and get predictions

---

## 👤 Author

**Rohan Sharma**
Aspiring Data Scientist | Python · Machine Learning · Data Analysis

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/rohan-sharma-8057b0331/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/RoHaN-0053)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
