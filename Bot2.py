import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("ArtificialIntelligence-Assessment.py//cybersecurity_intrusion_data.csv")

X = df[['failed_logins']]
y = df['attack_detected']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", accuracy)

sorted_idx = X_test['failed_logins'].argsort()
X_test_sorted = X_test.iloc[sorted_idx]
y_test_sorted = y_test.iloc[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.scatter(X_test_sorted, y_test_sorted, color='blue', label='Actual')
plt.scatter(X_test_sorted, y_pred_sorted, color='red', alpha=0.6, label='Predicted')
plt.plot(X_test_sorted, y_pred_sorted, color='red', linestyle='-', alpha=0.5)

plt.xlabel('Failed Logins')
plt.ylabel('Attack Detected')
plt.title('KNN Predictions vs Actual')
plt.legend()
plt.show()
