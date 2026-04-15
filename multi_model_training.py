import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/heart.csv")

# Features & Target
X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Store results
results = []

# Train & Evaluate
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, accuracy, precision, recall, f1])

# Display results
print("\nModel Comparison:\n")
print("Model | Accuracy | Precision | Recall | F1 Score")
print("-" * 55)

for r in results:
    print(f"{r[0]} | {r[1]:.2f} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f}")
    import matplotlib.pyplot as plt

# Extract data
model_names = [r[0] for r in results]
accuracies = [r[1] for r in results]
precisions = [r[2] for r in results]
recalls = [r[3] for r in results]
f1_scores = [r[4] for r in results]

# Plot
x = range(len(model_names))

plt.figure()
plt.bar(x, accuracies)
plt.xticks(x, model_names)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Models")

plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Use best model (Random Forest)
best_model = models["Random Forest"]

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Random Forest")
plt.show()
# -------------------------------
# FEATURE IMPORTANCE (Random Forest)
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# Get feature names
feature_names = X.columns

# Get importance
importances = best_model.feature_importances_

# Create DataFrame
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort
feat_df = feat_df.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(feat_df)

# Plot
plt.figure()
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.show()