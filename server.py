import os
import pickle
from typing import Optional, Tuple
from threading import Lock

from flask import Flask, request, jsonify

from app.query_data import get_chain
from app.ingest_data import create_vectorstore

app = Flask(__name__)

# # Load the vectorstore
# with open("app/vectorstore.pkl", "rb") as f:
#     vectorstore = pickle.load(f)

# Check if the vectorstore file exists, and if not, create it

def check():
    vectorstore_file = "vectorstore.pkl"
    
    create_vectorstore()

    with open(vectorstore_file, "rb") as f:
        vectorstore = pickle.load(f)
        return vectorstore

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

            # # Run chain and append input.
            # chain = get_chain(vectorstore)
            # print(chain)

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
    chain = get_chain(check())
    print(chain)
    if chain is None:
        return jsonify({"error": "Invalid OpenAI API key"}), 400

    chat = ChatWrapper()
    output,_ = chat(inp, history, chain)

    return {"output": output}

app.add_url_rule("/api/chat", "chat_api", chat_api, methods=["POST"])


# @app.route("/create_vectorstore", methods=["POST"])
# def create_vectorstore():

#     create_vectorstore()
#     # Return a response indicating success
#     return jsonify({"message": "Vector store created successfully!"}), 201

if __name__ == "__main__":
    app.run()
