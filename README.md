# AI-Based Disease Detection Through Symptoms

## Project Overview

The AI-Based Disease Detection Through Symptoms system is a web-based intelligent healthcare assistant designed to predict diseases using machine learning techniques. The system analyzes user-provided symptoms and generates a probable disease along with comprehensive health-related information, including precautions, medications, diet plans, and workout suggestions.
<p align="center">
  <img src="assets/home.png" width="700"/>
</p>
<p align="center"><em>Main Dashboard</em></p>
This project addresses a common problem in healthcare: early-stage disease identification. Many diseases share overlapping symptoms, which often leads to confusion and delayed diagnosis. This system provides a preliminary analysis tool that improves awareness and supports early decision-making.

The application is implemented using a Flask-based web framework and integrates a trained neural network model for prediction. It operates locally and can be extended for real-world deployment.

---

## Core Features

The system is designed with a focus on usability, accuracy, and practical healthcare support.

* Symptom-based disease prediction using a trained machine learning model
* Web-based user interface for easy interaction
* Detailed disease description for user awareness
* Preventive measures to reduce health risks
* Suggested medications (informational purpose only)
* Personalized diet recommendations
* Workout and physical activity suggestions
* Modular architecture for easy scalability

---

## System Architecture

The project follows a three-layer architecture:

### 1. Frontend Layer

The frontend is responsible for user interaction. It is built using HTML, CSS, and Flask templating.

Responsibilities:

* Accepting user input (symptoms)
* Displaying prediction results
* Showing additional health recommendations
* Providing a clean and responsive user interface

### 2. Backend Layer

The backend is implemented using Flask and acts as the core controller of the system.

Responsibilities:

* Handling HTTP requests and responses
* Loading trained machine learning models
* Processing user input
* Connecting datasets with prediction results
* Rendering dynamic web pages

### 3. Machine Learning Layer

This layer is responsible for disease prediction.

Responsibilities:

* Training the model on symptom-disease data
* Generating predictions based on input features
* Returning the most probable disease class

---

## Machine Learning Model

The system uses a Multi-Layer Perceptron (MLP) classifier, a type of artificial neural network suitable for classification tasks.

### Model Architecture

* Input Layer: Represents all possible symptoms as binary features
* Hidden Layer 1: 128 neurons with ReLU activation
* Hidden Layer 2: 64 neurons with ReLU activation
* Output Layer: Multi-class classification for diseases

### Training Details

* Dataset split: 70% training, 30% testing
* Label encoding applied to convert disease names into numeric form
* Optimizer: Adam
* Early stopping used to prevent overfitting
* Model saved using joblib for reuse

### Sample Code

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=500
)

model.fit(X_train, y_train)
```

---

## Dataset Description

The system relies on multiple structured datasets to provide both prediction and recommendations.

### Training Dataset

Contains binary values representing symptom presence or absence. Each row corresponds to a patient, and the target column represents the disease.

### Supporting Datasets

* symptoms_df.csv: Maps diseases to their symptoms
* description.csv: Provides detailed explanations of diseases
* precautions_df.csv: Contains preventive measures
* medications.csv: Lists commonly used medications
* diets.csv: Provides diet plans for diseases
* workout_df.csv: Suggests exercises and activities

### Example Code

```python
import pandas as pd

training_data = pd.read_csv("datasets/Training.csv")
description_data = pd.read_csv("datasets/description.csv")
```

---

## System Workflow

The system follows a structured pipeline from input to output:

1. User enters symptoms separated by commas
2. Input is preprocessed and matched with dataset features
3. A binary feature vector is generated
4. The trained model predicts the disease
5. The system retrieves related information from datasets
6. Results are displayed on the web interface

### Sample Logic

```python
symptoms = input_data.split(",")

input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]

prediction = model.predict([input_vector])
```

---

## Project Structure

The project is organized to ensure modularity and maintainability.

```
AI-Disease-Detection/
│
├── datasets/
│   ├── Training.csv
│   ├── symptoms_df.csv
│   ├── description.csv
│   ├── precautions_df.csv
│   ├── medications.csv
│   ├── diets.csv
│   └── workout_df.csv
│
├── templates/
│   ├── index.html
│   ├── result.html
│
├── static/
│   └── style.css
│
├── screenshots/
│
├── app.py
├── mlp_model.pkl
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AI-Disease-Detection.git
cd AI-Disease-Detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python app.py
```

Access the application at:
http://127.0.0.1:5000

---

## Screenshots

Place your project screenshots inside the `screenshots/` directory and reference them below.

```md
![Home Page](screenshots/home.png)
![Input Page](screenshots/input.png)
![Prediction Result](screenshots/result.png)
```

---

## Performance Evaluation

The model performance is evaluated using standard classification metrics:

* Accuracy: Measures overall correctness
* Precision: Measures correctness of positive predictions
* Recall: Measures ability to identify actual cases

The neural network demonstrates reliable performance for symptom-based disease prediction.

---

## Tools and Technologies

Programming Language:

* Python

Frameworks and Libraries:

* Flask
* pandas
* numpy
* scikit-learn
* joblib

Frontend:

* HTML
* CSS

Development Environment:

* Visual Studio Code
* Localhost deployment

---

## Deployment

The application is deployed as a local web application using Flask. It runs on a local server and can be accessed through a browser.

It can be extended to:

* Cloud deployment (AWS, Azure, Heroku)
* Mobile applications
* API-based architecture

---

## Limitations

* Predictions are based only on predefined datasets
* Not a substitute for professional medical advice
* Limited NLP capabilities
* No real-time clinical data integration

---

## Future Enhancements

* Integration with real-time healthcare APIs
* Mobile application development
* Advanced deep learning models
* Multilingual support
* Doctor consultation system
* Improved NLP for better interaction
---
## Contributors

* Ali Raza
* Waseem Abbas
---

## Conclusion

This project demonstrates the practical application of artificial intelligence in healthcare. By combining machine learning with a web-based interface, the system provides an accessible tool for early disease detection and health awareness.

It serves as a strong foundation for further development into a full-scale intelligent healthcare system.

---

