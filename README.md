# chef_gpt


Welcome to ChefGPT's repo, your friendly assistant for creating delicious holiday treats this season!

This chatbot works by taking in an input question (keyboard or mic) as an utterance from which it will detect an intent (with a certain degree of confidence) and depending on the intent, will pass on the utterance to an API (WolframAlpha) or a RAG recipe service
that will return an appropriate response.

The intent detection model is trained on the following out-of-scope intent classification dataset fount on kaggle: https://www.kaggle.com/datasets/stefanlarson/outofscope-intent-classification-dataset with 150 different intents (classes).

To provide a good user interface for the chatbot I created a HTML, CSS and JavaScript based web application.

To use ChefGPT simply dowload the 'jbruck_chatbot' folder or clone the repo and open the 'jbruck_chatbot' folder.
In your terminal run the following command:  pip install -r ./code/requirements.txt
Next, run the following command: python ./code/FlaskAgent.py

Copy the link that appears in the terminal and copy it to your browser (or ctrl+click on the link).
