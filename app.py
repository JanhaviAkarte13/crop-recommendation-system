#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import gradio as gr

#Load dataset
df = pd.read_csv("Crop_recommendation.csv") 

#Encode target labels (crop names) into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize ML models
naive_bayes_model = GaussianNB()
logistic_model = LogisticRegression(max_iter=300)
svm_model = SVC(probability=True)
random_forest_model = RandomForestClassifier(n_estimators=100)
knn_model = KNeighborsClassifier(n_neighbors=5)
decision_tree_model = DecisionTreeClassifier()

# Train models
naive_bayes_model.fit(X_train, y_train)
logistic_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)

# Define prediction function for Gradio
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall, model_choice):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    model_dict = {
        "Naive Bayes": naive_bayes_model,
        "Logistic Regression": logistic_model,
        "Support Vector Machine (SVM)": svm_model,
        "Random Forest": random_forest_model,
        "K-Nearest Neighbors (KNN)": knn_model,
        "Decision Tree": decision_tree_model
    }

    model = model_dict.get(model_choice)
    if model is None:
        return "⚠️ Invalid model selected"

    prediction = model.predict(input_data)
    predicted_crop = encoder.inverse_transform(prediction)[0]
    return f"🌾 Recommended Crop: {predicted_crop}"

#Create Gradio interface
inputs = [
    gr.Number(label="Nitrogen (N)"),
    gr.Number(label="Phosphorous (P)"),
    gr.Number(label="Potassium (K)"),
    gr.Number(label="Temperature (°C)"),
    gr.Number(label="Humidity (%)"),
    gr.Number(label="pH"),
    gr.Number(label="Rainfall (mm)"),
    gr.Dropdown(
        ["Naive Bayes", "Logistic Regression", "Support Vector Machine (SVM)",
         "Random Forest", "K-Nearest Neighbors (KNN)", "Decision Tree"],
        label="Choose Model"
    )
]

outputs = gr.Text(label="Prediction")

demo = gr.Interface(
    fn=recommend_crop,
    inputs=inputs,
    outputs=outputs,
    title="🌿 Smart Crop Recommendation System",
    description="""
## Welcome to the Smart Crop Recommendation System!

Enter the soil and weather parameters below and select a machine learning model to get a recommendation for the most suitable crop.

**Input Parameters:**
- Nitrogen (N), Phosphorous (P), Potassium (K) in soil
- Temperature, Humidity, and pH of the environment
- Rainfall amount

Choose your preferred model from the dropdown.
""",
    theme="soft"
)

demo.launch()  
