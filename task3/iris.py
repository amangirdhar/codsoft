import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv"
df = pd.read_csv(url)

# Explore the dataset
print(df.head())  # Display the first few rows of the DataFrame
print(df.info())  # Get information about the DataFrame, e.g., data types and missing values
print(df['species'].value_counts())  # Check the distribution of the target variable

# Split the data into features (X) and target variable (y)
X = df.drop('species', axis=1)
y = df['species']

# Encode the target variable to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict the target variable on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification Report:")
print(report)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
