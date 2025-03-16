# Loan Approval Data Analysis and Preprocessing

This project focuses on **Loan Approval Data Analysis and Preprocessing** to prepare the dataset for a machine learning classification task. The goal is to clean, explore, and preprocess the dataset efficiently for optimal model performance.

## **Table of Contents**
- [1. Importing Libraries](#1-importing-libraries)
- [2. Load Dataset](#2-load-dataset)
- [3. Data Exploration](#3-data-exploration)
- [4. Numerical Variable Analysis](#4-numerical-variable-analysis)
- [5. Data Preprocessing](#5-data-preprocessing)
- [6. Machine Learning Readiness](#6-machine-learning-readiness)
- [7. Summary](#7-summary)
- [8. Next Steps](#8-next-steps)

---

## **1. Importing Libraries**
The script imports essential libraries for:
- **Data Handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `missingno`
- **Statistical Analysis:** `scipy.stats`
- **Machine Learning:** `sklearn`, `imblearn`, `xgboost`

---

## **2. Load Dataset**
- Loads `loan_data_set.csv` into a DataFrame (`df`)
- Displays the first few rows (`df.head()`)
- Prints dataset shape (`df.shape`)

---

## **3. Data Exploration**
### **3.1 Loan ID Analysis**
- Checks unique Loan IDs using `df.Loan_ID.nunique()`

### **3.2 Categorical Column Analysis**
- Defines `analyze_categorical()`, a function to analyze categorical columns.
- Displays value counts, percentages, and missing values.
- Visualizes distributions using `seaborn.countplot()`.
- Analyzes key categorical variables:
  - **Gender, Married, Education, Self_Employed, Credit_History, Property_Area, Loan_Status, Loan_Amount_Term.**

---

## **4. Numerical Variable Analysis**
### **4.1 Descriptive Statistics**
- Computes summary statistics (`df.describe()`) for:
  - **ApplicantIncome, CoapplicantIncome, LoanAmount**

### **4.2 Data Distributions**
- **Histograms:** `seaborn.histplot()` for feature distributions.
- **Violin Plots:** `sns.violinplot()` to understand data spread.

### **4.3 Correlation Heatmap**
- Computes correlation (`df.corr()`).
- Visualizes using `sns.heatmap()`.

### **4.4 Categorical vs Categorical Analysis**
- Uses `pd.crosstab().plot(kind='bar')` for:
  - **Gender vs Married**
  - **Self-Employed vs Credit History**
  - **Property Area vs Loan Status**

### **4.5 Categorical vs Numerical Analysis**
- Uses `sns.boxplot()` to compare:
  - **Loan_Status vs ApplicantIncome**
  - **Loan_Status vs CoapplicantIncome**
  - **Loan_Status vs LoanAmount**

### **4.6 Numerical vs Numerical Analysis**
- **Scatter plot:** ApplicantIncome vs CoapplicantIncome.
- Computes **Pearson correlation** & **T-test** for significance.

---

## **5. Data Preprocessing**
### **5.1 Handling Missing Values**
- Checks missing values (`df.isnull().sum()`).
- **Imputation Strategies:**
  - **Categorical:** Fill with mode (`df[column].fillna(df[column].mode()[0])`).
  - **Numerical:** Fill with median (`df[column].fillna(df[column].median())`).

### **5.2 Encoding Categorical Variables**
- Uses **One-Hot Encoding (`pd.get_dummies()`)** to convert categorical variables into numerical format.

### **5.3 Outlier Detection & Removal**
- Uses **Interquartile Range (IQR)** to remove extreme outliers.
- Applies **log transformation (`np.log1p()`)** to reduce skewness.

### **5.4 Feature Selection**
- Splits dataset into:
  - **Features (`X`)** – all columns except "Loan_Status".
  - **Target Variable (`y`)** – "Loan_Status".

### **5.5 Handling Class Imbalance**
- Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to create a balanced dataset.

### **5.6 Data Normalization**
- Applies **Min-Max Scaling (0-1 normalization)** to numerical features.

---

## **6. Machine Learning Readiness**
The dataset is now clean and ready for machine learning models, including:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**

---

## **7. Summary**
- **Loaded & explored** the dataset.
- **Analyzed** categorical and numerical variables.
- **Handled** missing values, outliers, and class imbalance.
- **Encoded & normalized** features for ML.
- **Visualized** data insights with graphs.
- **Prepared** dataset for machine learning models.

---
