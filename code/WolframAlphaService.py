import requests


def getAnswer(query):
    # Wolfram Alpha API endpoint
    # url = "http://api.wolframalpha.com/v1/result"
    url = "http://api.wolframalpha.com/v1/spoken"

    # Parameters for the query
    params = {
        "appid": "QG759U-K96T398GRW",
        "i": query
    }
    # Make the API call
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        print("Error making API call: ", response.text)
        return "Please reformulate your question."


def main():
    # Example questions
    questions = [
        "what is spaghetti",
        "what is the weather in Ottawa",
        "random question"
    ]

    # Get answers for each question
    for question in questions:
        answer = getAnswer(question)
        print(f"Question: {question}\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
