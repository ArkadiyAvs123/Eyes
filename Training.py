import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # for saving the model

# Load features and labels saved from your previous pipeline
# For now, let's assume you have them as NumPy arrays features.npy and labels.npy

features = np.load('features.npy')  # shape (61, 320)
labels = np.load('labels.npy')      # shape (61,)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Initialize Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(clf, 'rf_eeg_classifier.joblib')
print("Model saved to rf_eeg_classifier.joblib")

