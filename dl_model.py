import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("data/heart.csv")

# Features & Target
X = data.drop("target", axis=1).values
y = data["target"].values

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling (IMPORTANT)
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to float32 (for TensorFlow stability)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# -----------------------------
# Build Neural Network
# -----------------------------
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=10,
    verbose=1
)

# -----------------------------
# Prediction
# -----------------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# -----------------------------
# Evaluation
# -----------------------------
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("ANN Accuracy:", round(acc, 4))
print("ANN Recall:", round(rec, 4))

# -----------------------------
# Save Model & Scaler
# -----------------------------
model.save("dl_model.h5")
pickle.dump(scaler, open("dl_scaler.pkl", "wb"))

print("DL model and scaler saved successfully!")