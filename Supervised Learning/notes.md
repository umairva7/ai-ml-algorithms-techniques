# Supervised Learning  

Supervised learning is a type of machine learning where the model is trained on a labeled dataset (input + output). The goal is to learn the mapping from inputs to outputs so the model can make predictions on unseen data.  

---

## 📘 Algorithms  

### 1. Linear Regression  
- Used for predicting continuous values.  
- Fits a straight line (`y = mx + c`) to minimize error between actual and predicted values.  

### 2. Logistic Regression  
- Used for classification problems (yes/no, spam/not spam).  
- Predicts probability using the sigmoid function.  

### 3. Decision Trees  
- Splits data based on feature conditions.  
- Easy to interpret but prone to overfitting.  

### 4. Random Forest  
- Ensemble of multiple decision trees.  
- Reduces overfitting and improves accuracy.  

### 5. Support Vector Machines (SVM)  
- Finds the optimal hyperplane to separate classes.  
- Works well in high-dimensional spaces.  

### 6. K-Nearest Neighbors (KNN)  
- Classifies data based on the majority of its nearest neighbors.  
- Simple but can be slow on large datasets.  

---

## 📊 Evaluation Metrics  

Evaluation metrics help us understand how well a supervised learning model performs.  

### Regression Metrics  
- **MAE (Mean Absolute Error):** Average of absolute differences between predicted and actual values.  
- **MSE (Mean Squared Error):** Squares the differences, penalizing large errors more.  
- **RMSE (Root Mean Squared Error):** Square root of MSE, interpretable in the same units as the target.  
- **R² (Coefficient of Determination):** Explains how much variance in the target is captured by the model.  

### Classification Metrics  
- **Accuracy:** Proportion of correct predictions out of total predictions.  
- **Precision:** Out of predicted positives, how many are truly positive (focus on minimizing false positives).  
- **Recall (Sensitivity):** Out of actual positives, how many are correctly identified (focus on minimizing false negatives).  
- **F1-Score:** Harmonic mean of precision and recall, useful when data is imbalanced.  

---

## 🔄 Cross Validation  

- Cross-validation helps ensure that the model generalizes well and isn’t just memorizing training data.  
- **k-Fold Cross Validation:** Split data into *k* parts. Train on *k-1* and test on the remaining, repeat *k* times, then average results.  
- Prevents overfitting and gives a more reliable estimate of model performance.  

---

## 🎯 Feature Selection  

Feature selection improves model efficiency, reduces overfitting, and enhances interpretability.  

- **Backward Elimination:** Start with all features and remove the least significant one step by step.  
- **Forward Selection:** Start with no features and add the most significant one step by step.  
- **LASSO (Least Absolute Shrinkage and Selection Operator):**  
  - A regression technique that applies **L1 regularization**.  
  - Shrinks less important feature weights to **zero**, effectively performing feature selection automatically.  

---

## 🛠 Libraries Commonly Used  

- **scikit-learn:** For regression, classification, feature selection, and cross-validation.  
- **Statsmodels:** For detailed statistical analysis and regression summaries.  
- **Matplotlib / Seaborn:** For data visualization.  
- **Pandas / Numpy:** For data manipulation and preprocessing.  

---

✅ This folder contains notes and practice notebooks on **Supervised Learning** including algorithms, evaluation metrics, feature selection, and validation techniques.  
