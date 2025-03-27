# train_lr.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv("data/house_prices_dataset_lr.csv")

# Prepare features and target
features = ["SquareFootage", "Bedrooms", "Bathrooms"]
X = df[features]
y = df["Expensive"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with improved parameters
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model properly
model_data = {
    "model": model,
    "features": features,
    "version": "1.0"
}
joblib.dump(model_data, "lr_model.joblib")