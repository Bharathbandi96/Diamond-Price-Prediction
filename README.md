# Diamond Price Prediction

## Project Overview
Diamonds are among the most valued gemstones, prized for their rarity, brilliance, and investment potential. Predicting diamond prices accurately is a complex problem due to multiple influencing factors such as **cut**, **color**, **clarity**, **carat weight**, and various dimensional properties.

In this project, a **machine learning–based regression model** is developed to predict diamond prices based on their characteristics. The project includes:
- **Exploratory data analysis (EDA)**
- **Feature engineering**
- **Dimensionality reduction**
- **Model training and evaluation** using multiple regression algorithms

The notebook implements **k-fold cross-validation** for performance evaluation and applies **dimensionality reduction (PCA & variance-based selection)** to improve prediction accuracy.

---

## Dataset Information

**Dataset:** `train.csv`  
**Number of Samples:** 193,573  
**Number of Features:** 11  

| Feature | Description |
|----------|--------------|
| `id` | Unique ID for each diamond sample |
| `carat` | Weight of the diamond |
| `cut` | Quality of the diamond cut |
| `color` | Diamond color grade (D–J) |
| `clarity` | Diamond clarity grade |
| `depth` | Height of the diamond divided by average girdle diameter |
| `table` | Width of the table as a percentage of the diameter |
| `x` | Length in mm |
| `y` | Width in mm |
| `z` | Height in mm |
| `price` | Price of the diamond (target variable) |

---

## Technologies and Libraries Used

| Category | Libraries |
|-----------|------------|
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` |
| **Deep Learning (optional)** | `tensorflow`, `keras` |
| **Metrics & Evaluation** | `r2_score`, `mean_squared_error` |

---

## Exploratory Data Analysis (EDA)

- Checked for **null values** and **duplicates** (none found)
- Visualized **categorical feature distributions** (`cut`, `color`, `clarity`)
- Performed **correlation analysis** using a heatmap
- Scaled numerical features using **StandardScaler**
- Applied **Ordinal Encoding** to convert categorical variables into numeric form

---

## Feature Engineering

### **1. Ordinal Encoding**
Categorical features (`cut`, `color`, `clarity`) were encoded numerically to enable model processing.

### **2. Feature Scaling**
Standardized all features using `StandardScaler` to ensure uniform contribution to model training.

### **3. Dimensionality Reduction**
Two methods were explored:
- **PCA (Principal Component Analysis):** Retained top 6 components contributing most to variance.
- **Variance-Based Selection:** Retained features with correlation > 0.1 with target variable.

Variance-based selection showed better model performance and was used for final training.

---

## Model Training and Evaluation

### **Train-Test Split**
Data was divided using `train_test_split` with:
- **Test size:** 33%
- **Random state:** 2311347

### **Cross-Validation**
K-Fold Cross-Validation was applied with:
- **Optimal K value:** 3 (determined using Linear Regression performance)

---

## Machine Learning Models Implemented

| Model | Description | Optimal Hyperparameters | Accuracy (R²) |
|--------|--------------|-------------------------|----------------|
| **Linear Regression** | Simple linear model for baseline comparison | CV = 3 | 0.91 |
| **Random Forest Regressor** | Ensemble of decision trees for improved prediction | `n_estimators = 64`, `max_depth = 9` | 0.97 |
| **Decision Tree Regressor** | Non-parametric tree-based regression | `max_depth = 9` | 0.97 |
| **Gradient Boosting Regressor** | Sequential boosting of weak learners | *[parameters tuned later]* | *To be evaluated* |
| **Support Vector Regressor (SVR)** | Regression with kernel-based learning | *[parameters tuned later]* | *To be evaluated* |
| **Lasso & Ridge Regression** | Regularized linear models | `alpha ∈ [0.1, 1.0]` | *To be evaluated* |
| **Neural Network (Keras)** | Simple deep learning regression model | EarlyStopping used | *To be evaluated* |

*(Results shown are based on variance-based dimensionality reduction.)*

---

## Dimensionality Reduction Comparison

| Method | Description | Accuracy (Linear Regression) |
|---------|--------------|-----------------------------|
| **PCA (Principal Component Analysis)** | Projects data into 6 principal components | 0.85 |
| **Variance-Based Selection** | Keeps features with correlation > 0.1 | **0.91** |
 **Chosen method:** Variance-based feature selection (better model generalization)

---

##  Evaluation Metrics

Model performance was assessed using:
- **R² Score (Coefficient of Determination)**
- **Mean Squared Error (MSE)**
- **Cross-Validation Mean Accuracy**

---

##  Results Summary

| Model | R² Score | Remarks |
|--------|-----------|----------|
| Linear Regression | 0.91 | Strong linear relationship |
| Decision Tree | 0.97 | High accuracy; prone to overfitting |
| Random Forest | 0.97 | Best performer; balanced bias-variance |
| PCA-based Models | 0.85 | Reduced performance due to information loss |

---

##  Future Enhancements

- Implement **hyperparameter tuning** using `GridSearchCV`
- Introduce **XGBoost** and **LightGBM** for improved accuracy
- Build an **interactive web interface** for live diamond price prediction
- Explore **deep learning regressors** for non-linear relationships

---



