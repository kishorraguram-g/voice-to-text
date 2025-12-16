from flask import Flask, jsonify
from flask_cors import CORS
import threading

from whisper_listener import start_listener, result_queue

app = Flask(__name__)

# ✅ CORS MUST BE HERE, BEFORE ROUTES
CORS(app, supports_credentials=True)

listener_thread = None

@app.route("/start", methods=["POST", "OPTIONS"])
def start_listening():
    global listener_thread
    if listener_thread is None or not listener_thread.is_alive():
        listener_thread = threading.Thread(
            target=start_listener, daemon=True
        )
        listener_thread.start()
        return jsonify({"status": "listening started"})
    return jsonify({"status": "already running"})

@app.route("/poll", methods=["GET"])
def poll_text():
    texts = []
    while not result_queue.empty():
        texts.append(result_queue.get())
    return jsonify({"results": texts})

if __name__ == "__main__":
    # ❗ ABSOLUTELY REQUIRED
    app.run(debug=False, use_reloader=False)
