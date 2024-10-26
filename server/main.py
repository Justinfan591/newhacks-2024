import os
import sys
import openai
import sounddevice as sd
import numpy as np
import wavio
from dotenv import load_dotenv
from gtts import gTTS
from playsound import playsound
import tempfile
import time

# Load environment variables from the .env file
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global variables
fs = 44100  # Sampling frequency
channels = 1  # Mono audio
dtype = 'int16'  # Data type

recording_data = []  # List to store recording data
stream = None  # To hold the stream object

def startRecord():
    """Start recording audio from the microphone."""
    global stream, recording_data

    recording_data = []  # Reset the recording data

    def callback(indata, frames, time, status):
        """Callback function to collect audio data."""
        if status:
            print(status)
        recording_data.append(indata.copy())

    stream = sd.InputStream(samplerate=fs, channels=channels, dtype=dtype, callback=callback)
    stream.start()
    print("Recording started.")

def stopRecord():
    """Stop recording and save the audio to a file."""
    global stream, recording_data
    if stream is not None:
        stream.stop()
        stream.close()
        stream = None
        print("Recording stopped.")
        # Concatenate the recorded data
        audio_data = np.concatenate(recording_data)
        # Save to a WAV file
        filename = 'recorded_audio.wav'
        wavio.write(filename, audio_data, fs, sampwidth=2)
        print("Audio saved to", filename)
        return filename
    else:
        print("No recording in progress.")
        return None

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

Please answer with 'this call is likely a' 'scam' or 'spam' or 'legitimate' and provide a brief and concise explanation.
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

def speak_text(text):
    """Convert text to speech and play it."""
    try:
        # Create a temporary file without keeping it open
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name  # Get the name of the temp file
        # Now the file is closed, and we can write to it
        # Generate the speech
        tts = gTTS(text=text, lang='en')
        # Save the speech to the temp file
        tts.save(temp_file)
        # Play the audio file
        playsound(temp_file)
        # Delete the temporary file
        os.remove(temp_file)
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")

def detect_spam_call(file_path):
    """Full pipeline: Transcribe and classify the audio file."""
    print("Transcribing audio...")
    transcribed_text = transcribe_audio(file_path)
    if transcribed_text is None:
        print("Failed to transcribe audio.")
        return

    print("Transcription complete.")
    print(f"Transcribed Text:\n{transcribed_text}\n")

    print("Classifying transcript...")
    classification = classify_transcript(transcribed_text)
    if classification is None:
        print("Failed to classify transcript.")
        return

    print("Classification complete.")
    print(f"\nClassification Result:\n{classification}")

    # Speak the classification result
    print("Speaking the classification result...")
    speak_text(classification)
    print("Done.")

if __name__ == '__main__':
    # Main function
    print("Starting recording...")
    startRecord()
    time.sleep(5)  # Record for 5 seconds
    stopRecord()
    # Proceed if the audio file is available
    audio_file_name = 'recorded_audio.wav'
    if os.path.exists(audio_file_name):
        # Run the spam call detection
        detect_spam_call(audio_file_name)
    else:
        print("No audio file to process. Exiting.")
