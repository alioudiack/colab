from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Charger les données
train = pd.read_excel('Donnnées_Projet_M2SID2023_2024.xlsx')
cat_features = train.select_dtypes(exclude=np.number).columns.tolist()
dfEncoded = train.copy()
for var in cat_features:
    if var == "Traitement":
        dfEncoded[var] = dfEncoded[var].map({"Thrombolyse": 1, "Chirurgie": 0})
    elif var == "Evolution":
        dfEncoded[var] = dfEncoded[var].map({"Vivant": 1, "Deces": 0})
    elif var == "SEXE":
        dfEncoded[var] = dfEncoded[var].map({"Homme": 1, "Femme": 0})
    else:
        dfEncoded[var] = dfEncoded[var].map({"NON": 1, "OUI": 0})
y = dfEncoded['Evolution']
X = dfEncoded.drop(columns=['Evolution'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'model.pkl')

app = Flask(__name__)
CORS(app)  # Activer les CORS
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        for var in cat_features:
            if var in data:
                if var == "Traitement":
                    data[var] = 1 if data[var] == "Thrombolyse" else 0
                elif var == "SEXE":
                    data[var] = 1 if data[var] == "Homme" else 0
                else:
                    data[var] = 1 if data[var] == "OUI" else 0
        column_names = list(X.columns)
        input_values = list(data.values())
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
