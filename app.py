from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

app = Flask(__name__)

# =========================
# Load Dataset
# =========================
dataset = pd.read_csv('datasets/Training.csv')

X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=20
)

model_path = 'mlp_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# =========================
# Load Extra Files
# =========================
description = pd.read_csv('datasets/description.csv')
precautions = pd.read_csv('datasets/precautions_df.csv')
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv('datasets/diets.csv')
workout = pd.read_csv('datasets/workout_df.csv')

# =========================
# Helper Function
# =========================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    ].values[0]
    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()
    return desc, pre, med, die, wrkout

# =========================
# Routes
# =========================
@app.route('/')
def home():
    return render_template('index.html', symptoms=X.columns, diseases=le.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms_text = request.form.get('symptoms')
    user_symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]

    input_vector = np.zeros(len(X.columns))
    for s in user_symptoms:
        if s in X.columns:
            input_vector[X.columns.get_loc(s)] = 1

    predicted_index = model.predict([input_vector])[0]
    disease = le.inverse_transform([predicted_index])[0]

    desc, pre, med, die, wrkout = helper(disease)

    return render_template(
        'result.html',
        disease=disease,
        desc=desc,
        precautions=pre,
        medications=med,
        diet=die,
        workout=wrkout,
        symptoms=user_symptoms
    )

@app.route('/disease/<disease_name>')
def disease_info(disease_name):
    desc, pre, med, die, wrkout = helper(disease_name)
    return render_template('disease.html', disease=disease_name, desc=desc, precautions=pre, medications=med, diet=die, workout=wrkout)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
