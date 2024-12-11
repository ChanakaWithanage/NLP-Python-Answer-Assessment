# Programming Answer Assessment API

A Flask-based API that evaluates programming answers using machine learning. The system processes code submissions and provides feedback on correctness.

## Project Structure

```
programming_assessment_api/
├── app.py              # Flask server implementation
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
├── model/
│   └── programming_answer_assessment_model.joblib
├── data/
│   ├── dataset.csv     # Original training data
│   └── processed_programming_dataset.csv
└── test.json          # Sample test file
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
    - Place your training data in `data/dataset.csv`
    - The dataset should have columns: 'Instruction', 'Input', 'Output'

4. Train the model:
```bash
python train_model.py
```
This will:
- Process the dataset
- Train the machine learning model
- Save the model as `model/programming_answer_assessment_model.joblib`
- Create `data/processed_programming_dataset.csv`

5. Start the Flask server:
```bash
python app.py
```
The server will run on `http://localhost:5000`

## API Endpoints

### Health Check
```bash
GET /health
```
Returns the status of the API and model loading state.

### Assess Answer
```bash
POST /assess
Content-Type: application/json

{
    "question": "Write a function to add two numbers",
    "user_answer": "def add(a, b):\n    return a + b"
}
```

Response format:
```json
{
    "correctness": 1,
    "feedback": "Excellent work! Your solution is correct."
}
```

## Testing the API

1. Create a test.json file with sample input:
```json
{
  "question": "Write a function to add two numbers",
  "user_answer": "def add(a, b):\n    return a + b"
}
```

2. Test using curl:
```bash
curl -X POST http://localhost:5000/assess -H "Content-Type: application/json" -d @test.json
```

3. Expected response:
```json
{
    "correctness": 1,
    "feedback": "Excellent work! Your solution is correct."
}
```

## Error Handling

The API includes error handling for:
- Missing required fields in request
- Model loading issues
- Invalid input formats
- Internal processing errors

Error responses include descriptive messages to help identify the issue.

## Dependencies

Key dependencies include:
- Flask
- Flask-CORS
- scikit-learn
- pandas
- numpy
- joblib

See `requirements.txt` for complete list and versions.