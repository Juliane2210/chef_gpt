from flask import Flask, render_template, request
from flask_cors import CORS
import json

# Intent service could be hosted remotely but for this project is hosted locally
from IntentService import getIntent

# The question resolution service could be hosted remotely or locally.

# QAVacationService is local
import QAVacationService

import RecipeService

from RAGRecipeService import RAGRecipeGenerator

# WolframAlpha is an example of remote access
import WolframAlphaService

app = Flask(__name__)

# Required for html page to interact with other sites
CORS(app)  # Enable CORS for all routes

generator = RAGRecipeGenerator()


@app.route('/')
def index():
    return render_template('index.html')

# Create endpoint to cater input query of Exploria


def isGreetingIntent(intent_classification):
    if (intent_classification == "greeting"):
        return True
    return False


def isRecipeIntent(intent_classification):
    allowed_classifications = ["recipe", "meal_suggestion",
                               "cook_time", "ingredient_substitution", "ingredient_list"]
    return intent_classification in allowed_classifications


def isLifestyleIntent(intent_classification):
    #
    # Make a list of all the intents that match to lifestyle
    # Travel, Movie, Cooking
    #
    allowed_classifications = ["travel_alert",
                               "travel_suggestion", "travel_notification", "international_visa", "movie"]

    return intent_classification in allowed_classifications


@app.route('/submit', methods=["GET", "POST"])
def processInputQuery():
    req = request.get_json()
    if req is not None and "msg" in req:
        utterance = req["msg"]
        #
        # The intent service could be remotely hosted and returns the intent classification with a confidence score.
        #
        intent_json = json.loads(getIntent(utterance))
        intent_classification = intent_json["intent"]
        confidence_score = intent_json["confidence"]

        if (confidence_score < 20):
            return "I didn't understand, please rephrase your question."

        response = f"I am {confidence_score} percent confident you asked about '{intent_classification}'"

        if (isRecipeIntent(intent_classification)):
            # answer = RecipeService.getRecipes(utterance)
            answer = generator.generate_recipe(utterance)
            response = response + ".  " + answer

        elif (isLifestyleIntent(intent_classification)):
            #
            # Specialized responses for custom category
            #
            answer = QAVacationService.getAnswer(utterance)
            response = response + ".  " + answer

        elif (isGreetingIntent(intent_classification)):
            #
            # Custom Greeting
            #
            response = "Hello this is Chef GPT.  Ask me any question.  I specialize in Christmas treats."
        else:
            #
            # General knowledge query will go to WolframAlpha
            #
            answer = WolframAlphaService.getAnswer(utterance)
            response = response + ".  " + answer

        return response
    else:
        return "Invalid input"


# Main to run the server on specific port
if __name__ == '__main__':
    app.run(debug=False, port=2412)
