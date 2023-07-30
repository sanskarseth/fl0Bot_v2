import os
import pickle
from typing import Optional, Tuple
from threading import Lock

from flask import Flask, request, jsonify
# from flask_cors import CORS

from app.query_data import get_chain

app = Flask(__name__)

# Load the vectorstore
with open("app/vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            chain = get_chain(vectorstore)
            output = chain({"question": inp, "chat_history": history})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

def chat_api():
    data = request.json
    api_key = os.environ.get("OPENAI_API_KEY")
    inp = data.get("question")
    history = data.get("history")
    chain = get_chain(vectorstore)
    if chain is None:
        return jsonify({"error": "Invalid OpenAI API key"}), 400

    chat = ChatWrapper()
    history, _ = chat(api_key, inp, history, chain)
    return jsonify({"history": history})

app.add_url_rule("/api/chat", "chat_api", chat_api, methods=["POST"])

if __name__ == "__main__":
    app.run()
