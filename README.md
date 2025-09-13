#🌿 **Smart Crop Recommendation System**

🌾 Project Overview

**The Smart Crop Recommendation System** is a machine learning-based project designed to help farmers and agricultural enthusiasts choose the most suitable crop for their soil and weather conditions. By analyzing key parameters like Nitrogen (N), Phosphorous (P), Potassium (K), Temperature, Humidity, pH, and Rainfall, the system predicts the best crop that can be cultivated.

To improve reliability, the project implements multiple machine learning models including:

Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Random Forest

K-Nearest Neighbors (KNN)

Decision Tree

Users can choose their preferred model and compare performances through accuracy scores and confusion matrices.

The project also includes a Gradio-based interactive web interface where users can input soil and weather values and instantly receive crop recommendations.

This system aims to:
Support sustainable farming

Maximize yield efficiency

Provide data-driven insights to farmers

✨ Features  
🌱 Predicts the best crop based on soil & weather conditions  
🧑‍🌾 User-friendly "Gradio Web App" 
📊 Performance comparison of multiple ML models  
🔎 Includes "EDA, Confusion Matrices & Accuracy Scores" 
⚡ Easy to deploy and run locally  


🛠 Tech Stack  
Programming Language:Python  
Libraries & Tools:  
  Pandas, NumPy, Scikit-learn  
  Matplotlib, Seaborn (for visualization)  
  Gradio (for deployment)  

📂 Dataset  
Source: Kaggle – [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  
Size: ~2,200 rows × 8 features  
Features: 
  N (Nitrogen), P (Phosphorous), K (Potassium)  
  Temperature, Humidity, pH, Rainfall  
  Label (Crop Name)  

> ⚠️ The dataset is not included in the repo. Please download it from Kaggle and place it in your project folder.  

🔄 Project Workflow 
A[Dataset] --> B[Data Preprocessing]
B --> C[Exploratory Data Analysis]
C --> D[Model Training]
D --> E[Model Evaluation]
E --> F[Deployment with Gradio]

⚙️ Installation & Setup

Clone this repository:

git clone https://github.com/your-username/crop-recommendation.git
cd crop-recommendation


Install dependencies:

pip install -r requirements.txt


Run the app:

python app.py


Open the link in your browser to access the Gradio interface.

🚀 Usage

Enter soil & weather parameters:

Nitrogen (N), Phosphorous (P), Potassium (K)

Temperature, Humidity, pH, Rainfall

Select a machine learning model from the dropdown

Click Submit to get crop recommendation


The output will display the best-suited crop 🌾
