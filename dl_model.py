import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv("data/heart.csv")

X = data.drop("target", axis=1)
y = data["target"]

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build ANN
model = Sequential()
model.add(Dense(16, activation='relu', input_dim=X.shape[1]))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Predict
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluate
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print("ANN Accuracy:", acc)
print("ANN Recall:", rec)

# Save model
model.save("dl_model.h5")