# E-Commerce Purchase Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green?logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange?logo=scikitlearn&logoColor=white)
![Deployment](https://img.shields.io/badge/Deployment-Render-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔍 Problem Statement
In e-commerce platforms, not every user who browses a product ends up purchasing it.  
The goal of this project is to **predict whether a user will purchase a product** based on their browsing behavior, demographics, and product details.

**Business Impact:**
- Identify high-intent users  
- Improve targeting and recommendations  
- Increase conversion rate  
- Optimize marketing spend  

---

## 📂 Dataset
- **Type:** Synthetic E-commerce User Behavior Data  
- **Target Variable:** `Purchased` (0 = No, 1 = Yes)  
- **Problem Type:** Binary Classification  

---

## Project Workflow (End-to-End)

### 1️⃣ Data Generation
- Synthetic dataset created to simulate real e-commerce user behavior  
- Includes demographic, browsing, and product features  

### 2️⃣ Data Understanding (EDA)
- Analyzed distributions of numerical features  
- Checked skewness and applied transformations  
- Visualized user behavior trends  
- Studied purchase vs non-purchase patterns  

### 3️⃣ Data Cleaning
- Handled missing values  
- Removed duplicate records  
- Ensured correct data types  
- Validated value ranges  

### 4️⃣ Feature Engineering
- Log transformation for skewed features  
- Square-root transformation for product price  
- Created derived features:
  - `Age_group` (Young / Middle / Senior)
  - `Price_range` (Cheap / Medium / Expensive)
  - `HighViewedTime` (0 / 1)

### 5️⃣ Feature Encoding & Scaling
- OneHotEncoding for categorical variables  
- StandardScaler for numerical features  
- Ensured same preprocessing pipeline for training and inference  

### 6️⃣ Train–Test Split
- Dataset split into training and testing sets  
- Maintained class balance  

---

## 🤖 Models Implemented

| Model | Purpose |
|------|--------|
| Logistic Regression | Baseline and interpretable model |
| K-Nearest Neighbors (KNN) | Distance-based comparison |
| Random Forest Classifier | Non-linear ensemble model |

---

## 📊 Model Evaluation
- Metrics used:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  
- Compared false positives and false negatives  
- Selected best model based on business-relevant performance  

---

## 🔍 Model Interpretation
- Analyzed Logistic Regression coefficients  
- Identified key drivers of purchase:
  - Viewed time
  - Product price range
  - Browsed category
  - Age group  

---

## 🚀 Deployment
- **FastAPI** used to build inference API  
- Model, scaler, and encoder saved using `joblib`  
- API deployed live on **Render**  
- Interactive Swagger docs available  

---

## 🌐 Frontend Integration
- HTML + JavaScript frontend  
- User inputs sent to FastAPI via POST request  
- Displays:
  - Purchase prediction
  - Purchase probability  

---

## 🛠️ Tech Stack
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.6.0-lightblue?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26.0-lightgrey?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange?logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-red?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-blue?logo=seaborn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green?logo=fastapi&logoColor=white)
![Render](https://img.shields.io/badge/Deployment-Render-brightgreen)
![Git](https://img.shields.io/badge/Git-2.42.0-orange?logo=git&logoColor=white)

---

## 🌟 Key Learnings
- Building ML intuition for feature engineering  
- Understanding skewness and transformations  
- Handling categorical data safely in production  
- End-to-end ML system design  
- API development and deployment  
- Debugging real production-level errors  
- Integrating ML models with frontend applications  

---

## 📌 Future Improvements
- Add product recommendation ranking  
- Implement session-based recommendations  
- Try advanced models (XGBoost, LightGBM)  
- Improve frontend UI/UX  
- Add user history and personalization  

---

✨ *This project represents a complete real-world Data Science workflow — from raw data to a deployed, production-ready machine learning system.*
