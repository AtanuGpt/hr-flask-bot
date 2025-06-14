from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv
import os
import uuid
import azure.cognitiveservices.speech as speechsdk
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, load_index_from_storage

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key'
PERSIST_DIR = "./storage"
WELCOME_MESSAGE = "Hello, I am your HR Bot. I will try my best to answer your question from my knowledge base."

# Text-to-Speech Function
def text_to_speech(text):
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")

    if not speech_key or not speech_region:
        return None

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    filename = f"static/audio/output_{uuid.uuid4()}.mp3"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=filename)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return f"/{filename}"
    else:
        return None

# Fetch response from vector DB
def fetchData(user_question):
    try:
        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
        response = query_engine.query(user_question)

        sources = []
        for node in response.source_nodes:
            meta = node.metadata
            if "file_name" in meta:
                sources.append(meta["file_name"])
        return str(response), sources
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
    response_text, sources = fetchData(user_message)
    audio_path = text_to_speech(response_text)
    session["chat_history"].append({"sender": "AI", "message": response_text})
    session.modified = True

    return jsonify({
        "user_message": user_message,
        "bot_response": response_text,
        "sources": sources,
        "audio": audio_path
    })

@app.route("/reset")
def reset_chat():
    session.clear()
    return '', 204

if __name__ == "__main__":
    if not os.path.exists("static/audio"):
        os.makedirs("static/audio")
    app.run(debug=True)
