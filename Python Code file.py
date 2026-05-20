import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================
dataset = pd.read_csv('datasets/Training.csv')

X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=20
)

# =========================
# Build & Train Model
# =========================
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

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# =========================
# Prediction Function
# =========================
def predict_disease(symptoms_vector):
    symptoms_vector = np.array(symptoms_vector).reshape(1, -1)
    predicted_index = model.predict(symptoms_vector)[0]
    return le.inverse_transform([predicted_index])[0]

# Example test
sample_symptoms = X_test.iloc[0].values
print("Predicted:", predict_disease(sample_symptoms))
print("Actual   :", le.inverse_transform([y_test[0]])[0])

# =========================
# Load Additional Datasets
# =========================
symp = pd.read_csv('datasets/symtoms_df.csv')
description = pd.read_csv('datasets/description.csv')
precautions = pd.read_csv('datasets/precautions_df.csv')
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv('datasets/diets.csv')
workout = pd.read_csv('datasets/workout_df.csv')

# =========================
# Fill Missing Values
# =========================
def fill(symp, group_col, fill_cols):
    for group_val in symp[group_col].unique():
        group_data = symp[symp[group_col] == group_val]
        for col in fill_cols:
            if col in symp.columns:
                mode_val = group_data[col].dropna().mode()
                fill_value = mode_val[0] if not mode_val.empty else symp[col].mode()[0]
                symp.loc[
                    (symp[group_col] == group_val) & (symp[col].isnull()), col
                ] = fill_value
    return symp

fill(symp, "Disease", ["Symptom_4"])

# =========================
# Helper Function
# =========================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    ].values.tolist()
    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()
    return desc, pre, med, die, wrkout

# =========================
# User Input Prediction
# =========================
symptoms = input("Enter your symptoms (comma-separated): ")
user_symptoms = [s.strip() for s in symptoms.split(',')]

input_vector = np.zeros(len(X.columns))
for symptom in user_symptoms:
    if symptom in X.columns:
        input_vector[X.columns.get_loc(symptom)] = 1

predicted_disease = predict_disease(input_vector)

desc, pre, med, die, wrkout = helper(predicted_disease)

# =========================
# Display Results
# =========================
print("\n================= Predicted Disease =================")
print(predicted_disease)

print("\n================= Description =======================")
print(desc)

print("\n================= Precautions =======================")
for i, p in enumerate(pre[0], 1):
    print(f"{i}. {p}")

print("\n================= Medications =======================")
for i, m in enumerate(med, 1):
    print(f"{i}. {m}")

print("\n================= Workout ===========================")
for i, w in enumerate(wrkout, 1):
    print(f"{i}. {w}")

print("\n================= Diet ==============================")
for i, d in enumerate(die, 1):
    print(f"{i}. {d}")
