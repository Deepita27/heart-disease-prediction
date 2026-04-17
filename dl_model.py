import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv("data/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to float (IMPORTANT)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Build model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Predict
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# Evaluate
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("ANN Accuracy:", acc)
print("ANN Recall:", rec)

# Save
model.save("dl_model.h5")
pickle.dump(scaler, open("dl_scaler.pkl", "wb"))

print("DL model and scaler saved!")