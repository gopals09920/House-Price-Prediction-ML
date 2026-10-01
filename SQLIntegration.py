import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Housing.csv")
conn = sqlite3.connect("Housing_database.db")
df.to_sql("housing_data", conn, if_exists="replace", index=False)
print("Data successfully SQL database me store ho gya!.")
query = "SELECT * FROM housing_data WHERE price IS NOT NULL"
data_from_db = pd.read_sql_query(query, conn)
print(data_from_db.head())
conn.close()
print("Missing value count:")
print(data_from_db.isnull().sum())
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    data_from_db[col] = data_from_db[col].map({'yes': 1, 'no': 0})
data_from_db = pd.get_dummies(data_from_db, columns=['furnishingstatus'], drop_first=True, dtype=int)
print("\nCleaned Data Preview:")
print(data_from_db.head())
X = data_from_db.drop(columns = ['price'])
Y = data_from_db['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
print("Model Successfully train ho gya!")
y_pred = model.predict(X_test)
accuracy = r2_score(Y_test, y_pred) * 100
print(f"\nModel Accuracy (R2 Score): {accuracy:.2f}%")
sample_house = X_test.iloc[[0]]
predicted_price = model.predict(sample_house)
actual_price = Y_test.iloc[0]
print(f"\nSample Prediction:")
print(f"Predicted Price: ₹{predicted_price[0]:.2f}")
print(f"Actual Price: ₹{actual_price:.2f}")
