from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np

app = Flask(__name__)

# Load your fish dataset (replace 'fish_data.csv' with your CSV file)
data = pd.read_csv('Fish.csv')

# Split the data into features (X) and labels (y)
X = data.drop('Species', axis=1)
y = data['Species']

# Train the Decision Tree classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the trained model to a .pkl file (e.g., 'fish_species_model.pkl')
joblib.dump(model, 'fish_species_model.pkl')


def predict_species(weight, length1, length2, length3, height, width):
    input_data = np.array([[weight, length1, length2, length3, height, width]])
    predicted_species = model.predict(input_data)[0]
    return predicted_species


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    weight = data.get('Weight')
    length1 = data.get('Length1')
    length2 = data.get('Length2')
    length3 = data.get('Length3')
    height = data.get('Height')
    width = data.get('Width')

    if None in [weight, length1, length2, length3, height, width]:
        return jsonify({"error": "Missing parameters. Please provide all 6 parameters."}), 400

    species = predict_species(weight, length1, length2, length3, height, width)

    return jsonify({"species": species}), 200


if __name__ == '__main__':
    app.run(debug=True)
