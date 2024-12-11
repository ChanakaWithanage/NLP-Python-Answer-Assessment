from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

class ProgrammingAnswerAssessment:
    def __init__(self):
        self.question_vectorizer = CountVectorizer()
        self.answer_vectorizer = CountVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def predict_answer(self, question, user_answer):
        # Vectorize input
        X_question = self.question_vectorizer.transform([question])
        X_answer = self.answer_vectorizer.transform([user_answer])
        X_input = np.hstack((X_question.toarray(), X_answer.toarray()))

        # Predict
        prediction = self.model.predict(X_input)
        correctness = self.label_encoder.inverse_transform(prediction)[0]

        # Generate feedback
        feedback = self.generate_feedback(correctness, question, user_answer)

        return correctness, feedback

    def generate_feedback(self, is_correct, question, user_answer):
        if is_correct:
            return "Excellent work! Your solution is correct."
        return "Your solution needs improvement. Please review and try again."

app = Flask(__name__)
CORS(app)

# Load the trained model
try:
    model = joblib.load('model/programming_answer_assessment_model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Initialize a new model instance instead of setting to None
    model = ProgrammingAnswerAssessment()
    print("Initialized new model instance")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/assess', methods=['POST'])
def assess_answer():
    try:
        data = request.get_json()

        if not data or 'question' not in data or 'user_answer' not in data:
            return jsonify({
                'error': 'Missing required fields. Please provide question and user_answer.'
            }), 400

        question = data['question']
        user_answer = data['user_answer']

        # Make prediction using the model
        correctness, feedback = model.predict_answer(question, user_answer)

        # Convert numpy int64 to regular Python int
        if isinstance(correctness, np.int64):
            correctness = int(correctness)

        return jsonify({
            'correctness': correctness,
            'feedback': feedback
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # This will print the full error in the console
        return jsonify({
            'error': f'Error processing request: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)