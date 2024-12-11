import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import difflib

def process_dataset(df):
    print("Processing dataset...")
    # Drop rows with missing Input
    df = df.dropna(subset=['Input'])

    # Create new columns with renamed columns
    processed_df = pd.DataFrame()
    processed_df['question'] = df['Instruction']
    processed_df['user_answer'] = df['Output']  # Initially use Output as user answer
    processed_df['is_correct'] = 1  # Assume correct initially

    # Simulate some incorrect answers (for demonstration)
    np.random.seed(42)
    incorrect_mask = np.random.random(len(df)) < 0.2  # 20% of answers will be marked incorrect
    processed_df.loc[incorrect_mask, 'is_correct'] = 0

    # For incorrect answers, slightly modify the code
    def introduce_error(code):
        lines = str(code).split('\n')
        if len(lines) > 2:
            error_line_index = np.random.randint(1, len(lines))
            lines[error_line_index] = '# Intentional error in this line'
        return '\n'.join(lines)

    processed_df.loc[incorrect_mask, 'user_answer'] = processed_df.loc[incorrect_mask, 'user_answer'].apply(introduce_error)

    print(f"Processed {len(processed_df)} rows")
    return processed_df

class ProgrammingAnswerAssessment:
    def __init__(self):
        self.question_vectorizer = CountVectorizer()
        self.answer_vectorizer = CountVectorizer()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def preprocess_data(self, df):
        # Vectorize questions and answers
        X_questions = self.question_vectorizer.fit_transform(df['question'])
        X_answers = self.answer_vectorizer.fit_transform(df['user_answer'])

        # Combine features
        X = np.hstack((X_questions.toarray(), X_answers.toarray()))

        # Encode labels
        y = self.label_encoder.fit_transform(df['is_correct'])

        return X, y

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

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

def main():
    # Load the original dataset
    print("Loading dataset...")
    df = pd.read_csv('data/dataset.csv')

    # Process the dataset
    processed_df = process_dataset(df)

    # Save processed dataset
    processed_df.to_csv('data/processed_programming_dataset.csv', index=False)
    print("Saved processed dataset")

    # Create and train the model
    print("Training model...")
    model = ProgrammingAnswerAssessment()
    X, y = model.preprocess_data(processed_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.train_model(X_train, y_train)

    # Save the model
    print("Saving model...")
    joblib.dump(model, 'model/programming_answer_assessment_model.joblib')
    print("Model saved successfully!")

    # Test the model
    print("\nTesting model with a sample...")
    sample_question = processed_df['question'].iloc[0]
    sample_answer = processed_df['user_answer'].iloc[0]
    correctness, feedback = model.predict_answer(sample_question, sample_answer)
    print(f"Sample prediction - Correctness: {correctness}, Feedback: {feedback}")

if __name__ == "__main__":
    main()