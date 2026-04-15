from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# -----------------------------
# Risk Classification Function
# -----------------------------
def classify_risk(prob):
    if prob < 30:
        return "Low Risk 😄"
    elif prob < 70:
        return "Medium Risk 😐"
    else:
        return "High Risk 😟"


# -----------------------------------
# Smart Medical Recommendation Engine
# -----------------------------------
def get_recommendations(prob, data):

    recommendations = []

    if prob < 30:
        message = "Your heart disease risk is low. Maintain a healthy lifestyle."
        recommendations.extend([
            "Maintain a balanced diet (fruits, vegetables, whole grains).",
            "Exercise at least 30 minutes daily.",
            "Avoid smoking and limit alcohol consumption.",
            "Maintain healthy body weight.",
            "Annual health check-up recommended."
        ])

    elif prob < 70:
        message = "You have moderate risk of heart disease. Lifestyle modifications and medical consultation are advised."
        recommendations.extend([
            "Reduce salt and processed food intake.",
            "Monitor blood pressure regularly.",
            "Control cholesterol levels.",
            "Practice stress management (yoga/meditation).",
            "Consult a doctor for further cardiac evaluation."
        ])

    else:
        message = "High probability of heart disease detected. Immediate medical consultation is strongly advised."
        recommendations.extend([
            "Consult a cardiologist as soon as possible.",
            "Avoid heavy physical exertion until medical evaluation.",
            "Strictly control diet (low salt, low saturated fat).",
            "Take prescribed medications regularly (if any).",
            "Seek emergency care if chest pain or breathlessness occurs."
        ])

    # Personalized Suggestions
    if data["chol"] > 240:
        recommendations.append("High cholesterol detected – reduce saturated fats and fried foods.")

    if data["trestbps"] > 140:
        recommendations.append("High blood pressure detected – monitor BP daily and reduce salt intake.")

    if data["exang"] == 1:
        recommendations.append("Exercise-induced angina present – avoid intense physical activities.")

    if data["oldpeak"] > 2:
        recommendations.append("Significant ST depression detected – cardiac evaluation recommended.")

    if data["ca"] >= 2:
        recommendations.append("Multiple vessels affected – specialist consultation required.")

    return message, recommendations


# -----------------------------------
# Feature Explanation Engine
# -----------------------------------
def explain_risk(data):

    explanations = []

    if data["age"] > 55:
        explanations.append("Advanced age increases cardiovascular risk.")

    if data["chol"] > 240:
        explanations.append("High cholesterol contributes to artery blockage.")

    if data["trestbps"] > 140:
        explanations.append("Elevated blood pressure increases heart workload.")

    if data["exang"] == 1:
        explanations.append("Exercise-induced angina suggests restricted blood flow.")

    if data["oldpeak"] > 2:
        explanations.append("Significant ST depression indicates abnormal stress response.")

    if data["ca"] >= 2:
        explanations.append("Multiple major vessels show signs of blockage.")

    if data["thal"] == 3:
        explanations.append("Abnormal thalassemia result associated with cardiac issues.")

    return explanations


@app.route("/")
def login():
    return render_template("login.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        data = {
            "age": float(request.form["age"]),
            "sex": float(request.form["sex"]),
            "cp": float(request.form["cp"]),
            "trestbps": float(request.form["trestbps"]),
            "chol": float(request.form["chol"]),
            "fbs": float(request.form["fbs"]),
            "restecg": float(request.form["restecg"]),
            "thalach": float(request.form["thalach"]),
            "exang": float(request.form["exang"]),
            "oldpeak": float(request.form["oldpeak"]),
            "slope": float(request.form["slope"]),
            "ca": float(request.form["ca"]),
            "thal": float(request.form["thal"])
        }

        df = pd.DataFrame([data])
        df_scaled = scaler.transform(df)

        probs = model.predict_proba(df_scaled)[0]

        disease_index = list(model.classes_).index(0)
        prob = probs[disease_index] * 100

        risk = classify_risk(prob)

        message, recommendations = get_recommendations(prob, data)
        explanations = explain_risk(data)

        return render_template(
            "result.html",
            probability=round(prob, 2),
            risk=risk,
            message=message,
            recommendations=recommendations,
            explanations=explanations
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)