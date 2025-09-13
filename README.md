# 🌿 Smart Crop Recommendation System   <!-- H1 -->

## 📌 Project Overview                  <!-- H2 -->

**The Smart Crop Recommendation System** is a machine learning-based project designed to help farmers and agricultural enthusiasts choose the most suitable crop for their soil and weather conditions. By analyzing key parameters like Nitrogen (N), Phosphorous (P), Potassium (K), Temperature, Humidity, pH, and Rainfall, the system predicts the best crop that can be cultivated.

To improve reliability, the project implements ** multiple machine learning models** including:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Decision Tree
Users can choose their preferred model and compare performances through accuracy scores and confusion matrices.The project also includes a Gradio-based interactive web interface where users can input soil and weather values and instantly receive crop recommendations.

This system aims to:  
- ✅ Support sustainable farming  
- ✅ Maximize yield efficiency  
- ✅ Provide data-driven insights to farmers  


## ✨ Features  
- 🌱 Predicts the best crop based on soil & weather conditions  
- 🧑‍🌾 User-friendly **Gradio Web App**  
- 📊 Performance comparison of multiple ML models  
- 🔎 Includes **EDA, Confusion Matrices & Accuracy Scores**  
- ⚡ Easy to deploy and run locally  

---

## 🛠 Tech Stack  
- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Pandas, NumPy, Scikit-learn  
  - Matplotlib, Seaborn (for visualization)  
  - Gradio (for deployment)  

---

## 📂 Dataset  
- **Source:** Kaggle – Crop Recommendation Dataset(https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)  
- **Size:** ~2,200 rows × 8 features  
- **Features:**  
- **Size:** ~2,200 rows × 8 features  
- **Features:**  
  - N (Nitrogen), P (Phosphorous), K (Potassium)  
  - Temperature, Humidity, pH, Rainfall  
  - Label (Crop Name)  

> ⚠️ The dataset is **not included** in this repository.  
> Please download it from Kaggle and place it in your project folder before running the code.  

---


## ➡️ Project Workflow

Dataset --> Data Preprocessing --> EDA --> Model Training --> Model Evaluation --> Deployment (Gradio)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository:

```bash
git clone [https://github.com/your-username/crop-recommendation.git](https://github.com/your-username/crop-recommendation.git)
cd crop-recommendation

**### 2. Install Dependencies:**
pip install -r requirements.txt

**### 3.Run the app:**
python app.py







