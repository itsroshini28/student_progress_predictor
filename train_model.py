
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle

# Load dataset
data = pd.read_csv("student_data.csv")

# Input features
X = data.drop("final_marks", axis=1)

# Target variable
y = data["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

# Calculate error
error = mean_absolute_error(y_test, predictions)

print("Model Training Completed!")
print("Mean Absolute Error:", error)

# Save model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
