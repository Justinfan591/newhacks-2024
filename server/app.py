from flask import Flask, render_template, request, jsonify
import os
import openai
import numpy as np
from dotenv import load_dotenv
from gtts import gTTS
import tempfile
import soundfile as sf

from flask import Flask
app = Flask(__name__)


# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio upload and processing
@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Get the audio file from the request
        audio_file = request.files['audio_data']
        audio_path = os.path.join('recorded_audio', 'uploaded_audio.wav')
        audio_file.save(audio_path)

        # Process the audio
        result = detect_spam_call(audio_path)

        # Return the result as JSON
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI's Whisper API."""
    try:
        with open(file_path, 'rb') as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        return transcript['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def classify_transcript(transcribed_text):
    """Classify the transcript using GPT-3.5 Turbo."""
    prompt = f"""
You are an assistant that classifies phone call transcripts as 'scam' or 'spam' or 'legitimate'.

Transcript:
\"\"\"
{transcribed_text}
\"\"\"

Please answer with 'This call is likely a scam', 'spam', or 'legitimate', and provide a brief and concise explanation.
"""
    try:
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'user', 'content': prompt}
            ],
            temperature=0,
            max_tokens=100,
            n=1,
            stop=None,
        )
        classification = response['choices'][0]['message']['content']
        return classification.strip()
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

def detect_spam_call(file_path):
    """Full pipeline: Transcribe and classify the audio file."""
    transcribed_text = transcribe_audio(file_path)
    if transcribed_text is None:
        return "Failed to transcribe audio."

    classification = classify_transcript(transcribed_text)
    if classification is None:
        return "Failed to classify transcript."

    return classification

if __name__ == '__main__':
    app.run(debug=True)
