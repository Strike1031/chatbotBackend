from flask import Flask, render_template, request, jsonify
from bot import get_response
from flask_cors import CORS
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

myport =  int(os.environ.get("PORT", 5000))

@app.route('/')
def message():
    return "Hey there. Groot's house"

@app.route('/predict', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input')
    user_id = data.get('id')
    bot_response = get_response(user_id, user_input)
    return {'bot_response': bot_response}

if __name__ == '__main__':
    app.run(port=myport)