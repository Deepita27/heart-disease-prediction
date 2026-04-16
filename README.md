# ❤️ Heart Disease Prediction System

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early prediction using clinical data can significantly improve patient outcomes. This project aims to develop a Machine Learning + Deep Learning based system to predict heart disease risk.

---

## 🔄 System Pipeline

Data Collection → Data Preprocessing → Feature Scaling →
ML Models (Logistic, Decision Tree, Random Forest) →
DL Model (ANN) → Prediction → Risk Classification → Recommendations

---

## 🚀 Features

* Predicts heart disease probability
* Risk classification (Low / Medium / High)
* Personalized recommendations
* Explainable AI (feature importance)
* Web-based user interface

---

## 🧠 Models Used

### 🔹 Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest (Best ML Model)

### 🔹 Deep Learning Model

* Artificial Neural Network (ANN) using TensorFlow/Keras

---

## 📊 Dataset

* UCI Cleveland Heart Disease Dataset
* 303 samples, 13 features
* Includes attributes like age, cholesterol, ECG results, etc.

---

## ⚙️ Tech Stack

* Python
* Flask
* Scikit-learn
* TensorFlow/Keras
* HTML, CSS, Bootstrap

---

## 📁 Project Structure

```
├── app.py
├── model.pkl
├── dl_model.h5
├── scaler.pkl
├── templates/
├── static/
├── data/
├── requirements.txt
```

---

## ▶️ How to Run the Project (IMPORTANT)

```bash
git clone https://github.com/Deepita27/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
python app.py
```

Open browser:

```
http://127.0.0.1:5000/
```

---

## 📦 Dependencies

* Flask
* scikit-learn
* pandas
* numpy
* tensorflow
* gunicorn

---

## 📈 Model Performance Comparison

| Model               | Accuracy | Recall |
| ------------------- | -------- | ------ |
| Logistic Regression | 85%      | 84%    |
| Random Forest       | 85%      | 88%    |
| ANN (Deep Learning) | 87%      | 90%    |

---

## 🧠 Justification of DL Model

The ANN model captures complex non-linear relationships in medical data, leading to improved recall and prediction performance compared to traditional ML models.

---

## ⚙️ Optimization Techniques

* Hyperparameter tuning
* Epoch tuning
* Feature scaling (StandardScaler)

---

## 📸 Sample Output

(Add screenshots of input form and result page here)

---

## ⚠️ Disclaimer

This system is a decision-support tool and not a replacement for professional medical diagnosis.

---

## 👩‍💻 Team Members

* Deepita S V
* Dharun Rahav M
* Hari Prasath P
* Kaushik S

---

## ⭐ Future Enhancements

* SHAP Explainability
* Cloud deployment
* Mobile application
* Integration with hospital systems
