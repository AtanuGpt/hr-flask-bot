from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv
import os
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'Pa55w0rd@123'  # Replace with a secure value
PERSIST_DIR = "./storage"
WELCOME_MESSAGE = "Hello, I am your HR Bot. I will try my best to answer your question from my knowledge base."

# Check for small talk
def is_small_talk(message):
    message = message.lower().strip()
    small_talk_phrases = ["hi", "hello", "hey", "thank you", "thanks", "bye", "goodbye", "start", "reset"]
    return any(phrase in message for phrase in small_talk_phrases)

# Fetch response and citations
def fetchData(user_question):
    if is_small_talk(user_question):
        if "thank" in user_question:
            return "You're welcome! 😊", []
        elif "bye" in user_question or "goodbye" in user_question:
            return "Goodbye! Have a great day!", []
        elif "start" in user_question:
            return WELCOME_MESSAGE, []
        else:
            return "I'm here to help. How can I assist you today?", []

    try:
        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=PERSIST_DIR
        )
        index = load_index_from_storage(storage_context=storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(user_question)

        sources = []
        for node in getattr(response, 'source_nodes', []):
            meta = node.node.metadata
            if "file_name" in meta:
                sources.append(meta["file_name"])

        return str(response), list(set(sources))

    except Exception as e:
        return f"Error: {str(e)}", []

@app.route("/")
def chat():
    session["chat_history"] = [{"sender": "AI", "message": WELCOME_MESSAGE}]
    session.modified = True
    return render_template("chat.html", chat_history=session["chat_history"])

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if "chat_history" not in session:
        session["chat_history"] = [{"sender": "AI", "message": WELCOME_MESSAGE}]

    session["chat_history"].append({"sender": "Human", "message": user_message})

    # Add loading message
    session["chat_history"].append({"sender": "AI", "message": "Fetching..."})
    session.modified = True

    response_text, sources = fetchData(user_message)

    # Replace last message (loading) with actual response
    session["chat_history"][-1] = {"sender": "AI", "message": response_text}
    session.modified = True

    return jsonify({
        "user_message": user_message,
        "bot_response": response_text,
        "sources": sources
    })

@app.route("/reset")
def reset_chat():
    session.clear()
    return '', 204

if __name__ == "__main__":
    app.run(debug=True)
