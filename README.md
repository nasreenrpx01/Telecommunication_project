# **Churn Prediction Model**

## **Overview**  
Hey there! 👋 Welcome to the **Churn Prediction Model** project. The primary goal is to predict which customers are likely to churn (leave) a telecommunications company. Using machine learning, we analyze customer behavior to help businesses take proactive steps to retain customers. In this project, we built, evaluated, and compared several models to determine the most accurate churn prediction system.

---

## **What’s Inside?**
1. **Data Exploration**: Understanding the dataset, identifying key features, and preparing data for modeling.
2. **Model Building**: Creating and training various machine learning models, including:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - Gradient Boosting
   - Support Vector Classifier (SVC)
   - XGBoost
3. **Evaluation**: Comparing models using metrics like accuracy, precision, recall, F1-score, and ROC AUC.  
4. **Deployment**: Deploying the best-performing model using Streamlit for real-time predictions.

---

## **The Dataset**  
The dataset contains detailed customer information, including demographics, account details, usage patterns, and the target variable, *churn*. Here's a quick look at the features:  

| **Feature**       | **Description**                                              |  
|--------------------|--------------------------------------------------------------|  
| `CustomerID`       | Unique identifier for each customer                          |  
| `Account Length`   | Number of days the customer has been with the company        |  
| `State`            | Customer's state (e.g., KS, CA)                              |  
| `Area Code`        | Area code of the customer                                    |  
| `Voice Plan`       | Whether the customer has a voice plan (Yes/No)               |  
| `Intl Plan`        | Whether the customer has an international plan (Yes/No)      |  
| `Usage Stats`      | Details about calls and minutes (day, evening, night, intl.) |  
| `Customer Calls`   | Number of calls made to customer service                     |  
| `Churn`            | Target variable (1 = Churn, 0 = No Churn)                    |  

---

## **Getting Started**  
To run the project locally, ensure you have the following installed:  
- **Python 3.x**  
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, and `streamlit`.

You can install them using:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
```

### **Project Structure**  
```
├── data/                 # Contains the dataset (e.g., churn_data.csv)
├── notebooks/            # Jupyter Notebooks for exploratory analysis and modeling
├── src/                  # Python scripts for data preprocessing and model building
├── deployment/           # Streamlit app scripts for deployment
├── README.md             # You're reading it now!
```

---

## **Data Preprocessing**  
Before training the models, the following steps were performed:  
1. **Handle Missing Values**: Fill or drop missing data.  
2. **Encode Categorical Variables**: Convert non-numeric columns (e.g., `State`, `Voice Plan`, `Intl Plan`) into numerical format.  
3. **Feature Scaling**: Normalize numerical columns to bring them onto the same scale.  

---

## **Model Building**  
We experimented with various machine learning models to predict churn. Here’s an example of training a **Random Forest Classifier**:  
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting the data
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
```

### **Model Evaluation Metrics**  
- **Accuracy**: Percentage of correct predictions.  
- **Precision**: Percentage of correctly predicted positive cases.  
- **Recall**: Percentage of actual positives correctly identified.  
- **F1-Score**: Harmonic mean of precision and recall.  
- **ROC AUC**: Ability to distinguish between churn and no churn.  

---

## **Model Results**  
Here’s how the models performed on the test data:  

| **Model**              | **Accuracy** | **ROC AUC** |  
|-------------------------|--------------|-------------|  
| Logistic Regression     | 91.59%       | 0.8513      |  
| Decision Tree           | 91.71%       | 0.8500      |  
| Random Forest           | 95.91%       | 0.8882      |  
| Gradient Boosting       | 96.03%       | 0.8853      |  
| SVC                     | 92.55%       | 0.8844      |  
| **XGBoost**             | **96.63%**   | **0.9008**  |  

**Best Model**: The **XGBoost Classifier** stood out as the best performer, achieving:  
- **Accuracy**: 96.63%  
- **ROC AUC**: 0.9008  

---

## **Visualization**  
### **Model Comparison**  
```python
import matplotlib.pyplot as plt

models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "SVC", "XGBoost"]
roc_auc_scores = [0.8513, 0.8500, 0.8882, 0.8853, 0.8844, 0.9008]

plt.figure(figsize=(10, 6))
plt.barh(models, roc_auc_scores, color='purple')
plt.xlabel("ROC AUC Score")
plt.ylabel("Models")
plt.title("Model Performance Comparison")
plt.show()
```

### **ROC Curves**  
The ROC curve for XGBoost showed the best results with an AUC of **0.9008**.

---

## **Deployment**  
The project is deployed as a web app using **Streamlit**, enabling real-time churn prediction.

### **Deployment Interface**  
The user can:
1. Upload a customer dataset to view predictions in bulk.
2. Input individual customer details using dropdowns and numeric inputs to predict churn.

Here’s the main deployment code:  
```python
import streamlit as st
import pandas as pd

st.title("Churn Prediction Model")

# Upload file or manual input
uploaded_file = st.file_uploader("Upload Data File")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

# Predict Button
if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_df)
        st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
```

---

## **Conclusion**  
- **Top Performer**: XGBoost was the best model, with **96.63% accuracy** and **ROC AUC = 0.9008**.  
- **Impact**: This model can help telecom companies identify at-risk customers and take preventive actions to retain them.  

---

## **Next Steps**  
1. **Hyperparameter Tuning**: Further optimize model parameters.  
2. **Deep Learning**: Experiment with neural networks for more advanced predictions.  
3. **Feature Engineering**: Add more customer-related features to improve model accuracy.  

That’s a wrap! 🚀 Feel free to explore, improve, or use this project. 😊
