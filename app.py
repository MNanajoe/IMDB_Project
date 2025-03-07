from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open("random_forest_sentiment.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

# Load vectorizer (if used during training)
vectorizer = TfidfVectorizer()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        review = data.get("review")
        if not review:
            return jsonify({"error": "Review text is required"}), 400
        
        # Transform input using vectorizer (assuming it was used in training)
        review_transformed = vectorizer.transform([review])
        
        # Predict sentiment
        prediction = model.predict(review_transformed)[0]
        
        # Return response
        return jsonify({"review": review, "sentiment": "positive" if prediction == 1 else "negative"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
