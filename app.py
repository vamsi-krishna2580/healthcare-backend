from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
import warnings
import os
import sys
import math
import json
import random

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


warnings.filterwarnings("ignore")

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = joblib.load(resource_path("model/svm_diabetes_model.pkl"))
scaler = joblib.load(resource_path("model/scaler.pkl"))
X_test = joblib.load(resource_path("model/X_test.pkl"))
y_test = joblib.load(resource_path("model/y_test.pkl"))

FEATURE_NAMES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data"}), 400

        print("\n=== Incoming request from frontend ===")
        print(json.dumps(data, indent=2))

        gender = str(data.get("gender", "male")).lower()
        pregnancies = float(data.get("pregnancies", 0)) if gender == "female" else 0.0

        glucose = max(0, min(200, float(data.get("glucose", 0))))
        blood_pressure = max(0, min(180, float(data.get("bloodPressure", 0))))
        skin_thickness = max(0, min(100, float(data.get("skinThickness", 0))))
        insulin = max(0, min(900, float(data.get("insulin", 0))))
        bmi = max(10, min(60, float(data.get("bmi", 10))))
        dpf = max(0.0, min(3.0, float(data.get("dpf", 0))))
        age = max(18, min(90, float(data.get("age", 18))))

        feature_values = [pregnancies,glucose,blood_pressure,
                          skin_thickness,insulin,bmi,dpf,age]

        print("Processed feature vector:")
        for n,v in zip(FEATURE_NAMES, feature_values):
            print(f"{n}: {v}")

        X = pd.DataFrame([feature_values], columns=FEATURE_NAMES)
        Xs = scaler.transform(X)

        pred = model.predict(Xs)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(Xs)[0][1]
        else:
            decision = model.decision_function(Xs)[0]
            prob = 1/(1+math.exp(-decision))

        risk_score = round(prob*100,2)
        label = "High Risk" if pred==1 else "Low Risk"

        print(f"Prediction: {label} | Risk: {risk_score}%")
        print("======================================\n")

        return jsonify({"prediction":label,"risk_score":risk_score,"probability":round(prob,4)})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error":"Prediction failed"}),500


@app.route("/population-stats", methods=["GET"])
def population_stats():
    data=[]
    for _ in range(500):
        age=random.randint(20,80)
        bmi=round(random.uniform(18,45),1)
        glucose=random.randint(70,200)
        bp=random.randint(60,160)

        has_diabetes = glucose>140 or bmi>32 or (age>55 and glucose>120)

        data.append({
            "age":age,
            "bmi":bmi,
            "glucose":glucose,
            "bloodPressure":bp,
            "hasDiabetes":bool(has_diabetes)
        })
    return jsonify(data)


# 🔹 NEW: model performance endpoint
@app.route("/model-metrics", methods=["GET"])
def model_metrics():
    try:
        threshold = float(request.args.get("threshold", 0.5))

        # Scale test data
        X_test_scaled = scaler.transform(X_test)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:,1]
        else:
            decision = model.decision_function(X_test_scaled)
            y_prob = 1/(1+np.exp(-decision))


        # Apply threshold manually
        y_pred = (y_prob >= threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Metrics
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        roc_points = [
            {"fpr": float(f), "tpr": float(t)}
            for f, t in zip(fpr, tpr)
        ]

        return jsonify({
            "confusion":{
                "truePositive": int(tp),
                "falsePositive": int(fp),
                "trueNegative": int(tn),
                "falseNegative": int(fn),
                "accuracy": round(accuracy,4),
                "precision": round(precision,4),
                "recall": round(recall,4),
                "f1Score": round(f1,4)
            },
            "roc": roc_points,
            "auc": round(float(auc),4)
        })

    except Exception as e:
        print("Metrics error:", e)
        return jsonify({"error":"Metrics failed"}),500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"Backend running"}),200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
