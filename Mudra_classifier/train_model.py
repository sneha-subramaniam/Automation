import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('mudra_data.csv')
X = df.iloc[:, :-1] # Landmarks
y = df.iloc[:, -1]  # Mudra Name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save the model to use in the app
with open('mudra_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"Model trained with accuracy: {model.score(X_test, y_test)}")