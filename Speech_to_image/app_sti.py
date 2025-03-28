import os
import io
import base64
from flask import Flask, render_template, request
import speech_recognition as sr
from pydub import AudioSegment
import requests
from monsterapi import client
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
MONSTER_API_KEY = os.getenv("MONSTER_API_KEY")
if not MONSTER_API_KEY:
    raise Exception("MONSTER_API_KEY not set in .env file.")

app = Flask(__name__)

# Initialize MonsterAPI client and speech recognizer
monster_client = client(MONSTER_API_KEY)
recognizer = sr.Recognizer()

# Audio Processing Function

def convert_audio_to_wav(audio_bytes, original_format):
    
    try:
        input_audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=original_format)
        wav_io = io.BytesIO()
        input_audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None

def transcribe_audio(audio_io):
    
    try:
        with sr.AudioFile(audio_io) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="en-US")
            return text
    except sr.UnknownValueError:
        return "Speech recognition failed."
    except sr.RequestError:
        return "Speech recognition API error."
    except Exception as e:
        return f"Error during transcription: {e}"


# Image Generation Function via MonsterAPI

def generate_image_from_text(prompt):
    
    input_data = {
        'prompt': prompt,
        'negprompt': 'deformed, bad anatomy, disfigured, poorly drawn face',
        'samples': 1,
        'steps': 30,           
        'aspect_ratio': 'square',
        'guidance_scale': 6.0,
        'seed': 2414,
    }
    try:
        result = monster_client.generate('txt2img', input_data, timeout=60)
        if "output" in result and result["output"]:
            img_url = result["output"][0]
            r = requests.get(img_url, timeout=60)
            if r.status_code == 200:
                image_b64 = base64.b64encode(r.content).decode("utf-8")
                return image_b64
            else:
                return f"Error: Failed to download image. Status code: {r.status_code}"
        else:
            return "Error: No output from MonsterAPI."
    except Exception as e:
        return f"Error generating image: {e}"

# Flask Route

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    image_data = ""
    error = ""
    if request.method == "POST":
        #Audio file upload
        if "audio_file" in request.files and request.files["audio_file"].filename != "":
            file = request.files["audio_file"]
            ext = file.filename.rsplit(".", 1)[1].lower()
            audio_bytes = file.read()
            wav_io = convert_audio_to_wav(audio_bytes, ext) if ext != "wav" else io.BytesIO(audio_bytes)
            if wav_io:
                transcription = transcribe_audio(wav_io)
            else:
                error = "Audio conversion failed."
        #Live recording 
        elif request.form.get("live") == "record":
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio_data = recognizer.listen(source, timeout=10)
                    transcription = recognizer.recognize_google(audio_data, language="en-US")
            except Exception as e:
                error = f"Live recording failed: {e}"
        else:
            error = "No audio input provided."

        if transcription and "failed" not in transcription.lower():
            image_data = generate_image_from_text(transcription)
            if image_data.startswith("Error"):
                error = image_data
        else:
            error = transcription or error

    return render_template("index_sti.html", transcription=transcription, image_data=image_data, error=error)

if __name__ == "__main__":
    app.run(debug=True)
