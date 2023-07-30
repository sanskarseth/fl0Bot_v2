import os
import pickle
from typing import Optional, Tuple
from threading import Lock

from flask import Flask, request, jsonify

from app.query_data import get_chain

app = Flask(__name__)

# Load the vectorstore
with open("app/vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(
        self, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Bad OpenAI key"))
                return history, history

            # Run chain and append input.
            chain = get_chain(vectorstore)
            print(chain)

            output = chain({"question": inp, "chat_history": history})["answer"]
            print(output)

            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

def chat_api():
    data = request.json
    inp = data.get("question")
    history = []
    chain = get_chain(vectorstore)
    print(chain)
    if chain is None:
        return jsonify({"error": "Invalid OpenAI API key"}), 400

    chat = ChatWrapper()
    output,_ = chat(inp, history, chain)

    return {"output": output}

app.add_url_rule("/api/chat", "chat_api", chat_api, methods=["POST"])

if __name__ == "__main__":
    app.run()
