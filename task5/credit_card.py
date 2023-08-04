import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Load the data
df = pd.read_csv("creditcard.csv")

# Normalize the features (except 'Time' and 'Class')
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Logistic Regression
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_normalized_df, y, test_size=0.2, random_state=25)
lm = LogisticRegression()
lm.fit(X_train_lr, y_train_lr)
predictions_lr = lm.predict(X_test_lr)
print("Logistic Regression:")
print(confusion_matrix(y_test_lr, predictions_lr))
print(classification_report(y_test_lr, predictions_lr))

# Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=25)
y_train_rf = y_train_rf.values.ravel()  # Convert the column vector to a 1-dimensional array
rfm = RandomForestClassifier(n_estimators=45, random_state=35)
rfm.fit(X_train_rf, y_train_rf)
predictions_rf = rfm.predict(X_test_rf)
print("Random Forest:")
print(confusion_matrix(y_test_rf, predictions_rf))
print(classification_report(y_test_rf, predictions_rf))
