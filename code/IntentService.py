import pandas as pd
import pickle
import warnings
import numpy as np
import json

warnings.filterwarnings('ignore')

# Load model
sgd = pickle.load(open('.//code//intent//models//SGD_model_inscope.pkl', 'rb'))
cv_in = pickle.load(
    open('.//code//intent//models//Count_vector_inscope.pkl', 'rb'))
encoder = pickle.load(
    open('.//code//intent//models//LabelEncoder_inscope.pkl', 'rb'))


def getIntent(utterance):
    # Convert data into dataframe
    data_df = pd.DataFrame({'text': [utterance]})
    X = cv_in.transform(data_df.text)

    # Predict
    result = sgd.predict(X)

    probabilities = sgd.predict_proba(X)
    confidence_score = max(probabilities[0])
    confidence_score = round(confidence_score*100)

    result = encoder.inverse_transform(result)

    result_dict = {
        "utterance": utterance,
        "intent": result[0],
        "confidence": confidence_score
    }
    # Convert the dictionary to JSON format
    result_json = json.dumps(result_dict)
    return result_json


def main():
    # Example calls to getIntent function
    intents = [
        "what is the cheapest vacation",
        "where to buy red t-shirt",
        "10-4",
        "how long should jello stay in the fridge",
        "how long will jello last",
        "what time is it",
        "how to cook jello",
        "what is the weather"
    ]

    for intent in intents:
        print(getIntent(intent))


if __name__ == "__main__":
    main()
