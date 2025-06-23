# Task 1: Advanced Data Cleaning and Preprocessing on the Ames Housing Dataset

## Overview
This project showcases a comprehensive data preprocessing pipeline on the **Ames Housing dataset**. Chosen for its complexity over the recommended Titanic dataset, this project demonstrates advanced techniques required for real-world data science tasks. The workflow covers in-depth missing value analysis, feature engineering, careful handling of various data types, and robust scaling, resulting in a dataset that is fully prepared for high-performance regression modeling.

## Project Scope
The objective of this task was to clean and prepare raw data for machine learning. This project successfully fulfills this by transforming the raw `train.csv` into a fully processed, feature-rich dataset, ready for modeling. **No model training was performed**, as it was outside the scope of the task.

## Files in this Repository
- `task1.ipynb`: The main Jupyter Notebook containing all Python code and step-by-step explanations.
- `train.csv`: The raw training data used for this project.
- `README.md`: This project summary and documentation.

## Advanced Preprocessing Steps Implemented
1.  **Context-Aware Missing Value Imputation:** Instead of generic imputation, missing values were handled based on their meaning in the data documentation. For example, `NaN` in `PoolQC` was correctly interpreted as "No Pool" and filled with 'None', while truly missing `LotFrontage` values were imputed using the median of their neighborhood.
2.  **Target Variable Transformation:** The highly skewed `SalePrice` was log-transformed (`np.log1p`) to approximate a normal distribution, a crucial step for many linear models.
3.  **Strategic Feature Engineering:** New, more informative features were created, such as `TotalSF` (total square footage) and `HouseAge`, to provide more predictive power to a future model.
4.  **Nuanced Categorical Encoding:** A distinction was made between ordinal and nominal data. Ordinal features (e.g., `ExterQual`) were manually mapped to preserve their inherent order, while nominal features were one-hot encoded to prevent false ordinal relationships.
5.  **Standardization:** All numerical features were scaled using `StandardScaler` to ensure they are on a comparable scale, which is vital for many ML algorithms.

## Interview Questions
*(Answers to the provided interview questions are included below)*

---

### **Interview Question Answers**

**1. What are the different types of missing data?**
   - **Missing Completely at Random (MCAR):** The missingness is unrelated to any other variable. It's pure chance.
   - **Missing at Random (MAR):** The missingness is related to another *observed* variable in the dataset.
   - **Missing Not at Random (MNAR):** The missingness is related to the value of the missing data itself.

**2. How do you handle categorical variables?**
   - **Label Encoding:** Assigns a unique integer to each category. Best for *ordinal* data where order matters (e.g., Low, Med, High).
   - **One-Hot Encoding:** Creates a new binary (0/1) column for each category. Best for *nominal* data where order does not matter (e.g., City).

**3. What is the difference between normalization and standardization?**
   - **Normalization (Min-Max Scaling):** Scales data to a fixed range, usually [0, 1]. Sensitive to outliers.
   - **Standardization (Z-score Normalization):** Scales data to a mean of 0 and a standard deviation of 1. It is less affected by outliers and is the more common choice.

**4. How do you detect outliers?**
   - **Visual Methods:** Using Box Plots and Scatter Plots to visually identify points far from the main data cluster.
   - **Statistical Methods:** Using the Interquartile Range (IQR) rule (e.g., points outside 1.5 * IQR) or Z-scores (e.g., points with a Z-score > 3).

**5. Why is preprocessing important in ML?**
   - It's crucial because real-world data is "dirty." Preprocessing improves model performance by fixing issues like missing values, inconsistent formats, and outliers. It ensures the model learns from accurate patterns, following the "Garbage In, Garbage Out" principle.

**6. What is one-hot encoding vs label encoding?**
   - **Label Encoding** creates a single column with numerical values (0, 1, 2...), which can imply a false order that misleads some models.
   - **One-Hot Encoding** creates multiple binary columns, which avoids this false order and is safer for nominal data.

**7. How do you handle data imbalance?**
   - **Resampling:** Either oversampling the minority class (e.g., with SMOTE) or undersampling the majority class.
   - **Using appropriate metrics:** Focusing on Precision, Recall, and F1-score instead of just accuracy.
   - **Algorithmic approaches:** Using models with class weight parameters.

**8. Can preprocessing affect model accuracy?**
   - **Absolutely, and profoundly.** Good preprocessing (like proper scaling and encoding) can dramatically increase model accuracy. Bad preprocessing (like dropping important features or using the wrong encoding method) can severely harm it. It is one of the most impactful stages of the ML pipeline.
