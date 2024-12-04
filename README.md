# **Churn Prediction Model**

## **Overview**
Hey there! 👋 Welcome to the Churn Prediction Model project. The goal here is pretty straightforward – to predict which customers might churn (leave) a telecommunications company. We use various machine learning models to understand customer behavior and help businesses take proactive steps to retain them. In this project, you'll see how we applied several models to predict churn and which one came out on top!

## **What’s Inside?**
- **Data Exploration**: We dived deep into the dataset to understand the features and what’s important for predicting churn.
- **Model Building**: We built and compared different machine learning models, such as Logistic Regression, Random Forest, and XGBoost.
- **Evaluation**: We evaluated each model using common metrics (accuracy, precision, recall, F1-score) to figure out which model gave the best results.
- **Best Model**: After a thorough comparison, we found that XGBoost was the winner, offering the best performance in terms of accuracy and F1-score.

## **The Dataset**
The dataset used for this project contains a bunch of customer-related information – everything from account details to usage statistics. Here's a quick peek at what the dataset includes:
- **CustomerID**: A unique identifier for each customer.
- **Account Length**: How long the customer has been with the company.
- **Service Type**: What type of service the customer uses.
- **Plan Type**: The subscription plan the customer has chosen.
- **Usage Stats**: Information about how much the customer uses the service (calls, data, etc.).
- **Churn**: The target variable (1 = churned, 0 = stayed).

## **Getting Started**

### **What You Need**
Before diving into the code, you'll need the following:
- **Python 3.x** (of course)
- Libraries like `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, and `plotly` (don't worry, I’ll show you how to install them)

You can quickly install everything by running:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly
```

### **How Everything is Organized**
Here’s a quick rundown of the project structure:
- **data/**: Contains the raw dataset (e.g., `churn_data.csv`).
- **notebooks/**: Where you’ll find step-by-step Jupyter notebooks documenting the whole process.
- **src/**: This folder has all the Python scripts for data processing, model training, and evaluation.
- **README.md**: The file you’re reading right now! 😉

### **Loading the Data**
Here’s how you load the data into Python using `pandas`:

```python
import pandas as pd
df = pd.read_csv("data/churn_data.csv")
```

### **Data Preprocessing**
Before jumping into building models, we need to clean the data:
- **Handle Missing Values**: Fill or drop missing values.
- **Encode Categorical Variables**: Turn non-numeric features into numbers.
- **Scale Features**: Normalize numerical values to make sure they're on the same scale.

### **Building Models**
We built several models to predict churn:
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
- **Gradient Boosting**
- **Support Vector Classifier (SVC)**
- **XGBoost**

Here’s an example of training a Random Forest model:

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
```

### **Model Evaluation**
To see how well each model performed, we used these metrics:
- **Accuracy**: How often the model made the right prediction.
- **Precision**: The percentage of correct positive predictions.
- **Recall**: The percentage of actual positives correctly identified.
- **F1-Score**: The balance between precision and recall.
- **ROC AUC**: How well the model distinguishes between the churned and non-churned customers.

### **Picking the Best Model**
After training and evaluating all models, the **XGBoost Classifier** stood out as the best performer. It had the highest accuracy and F1-score, which made it our top choice for predicting churn.

```python
print("Best Model: XGBClassifier")
```

### **Visualization**
I used **Plotly** to create some beautiful, interactive visualizations to show the model comparison. Here’s how you can visualize model accuracy:

```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(x=models, y=accuracies, orientation="h")])
fig.update_layout(title="Model Test Accuracies", xaxis_title="Accuracy", yaxis_title="Models")
fig.show()
```

## **Results**

Here’s how each model performed on the test set:
- **Logistic Regression**: 91.59%
- **Decision Tree**: 91.71%
- **Random Forest**: 95.91%
- **Gradient Boosting**: 96.03%
- **SVC**: 92.55%
- **XGBoost**: 96.63%

### **Top Performer: XGBoost**
- **Test Accuracy**: 96.63%
- **F1-Score (Macro Avg)**: 0.89
- **ROC AUC**: 0.97

## **Conclusion**
After evaluating different models, we found that **XGBoost** provides the best balance of accuracy and recall for churn prediction. It’s a reliable model to help businesses identify customers likely to churn and take action to retain them.

## **Next Steps**
While this project is already successful, there’s always room for improvement:
- **Hyperparameter Tuning**: Optimize model parameters for even better performance.
- **Deep Learning Models**: Explore neural networks for more advanced prediction techniques.
- **Additional Features**: Add more customer-related data, like feedback or customer support interactions, to further enhance the model.

---

That’s a wrap! I hope you enjoyed this walkthrough of the churn prediction project. Feel free to reach out if you have any questions or need more details. Happy coding! 😊
