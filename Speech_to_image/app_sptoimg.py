import os
import streamlit as st
import speech_recognition as sr
from monsterapi import client
import aiohttp
import asyncio
from PIL import Image
from dotenv import load_dotenv
from pydub import AudioSegment
from io import BytesIO

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("MONSTER_API_KEY")

if not api_key:
    st.error("API key not found. Make sure the .env file contains MONSTER_API_KEY.")
    st.stop()

monster_client = client(api_key)
recognizer = sr.Recognizer()

st.title("Speech-to-Image Generator")
st.write("Speak something in English, or upload an audio file to generate an image based on the audio!")

async def fetch_image_async(session, img_url):
    try:
        async with session.get(img_url, timeout=60) as response:
            if response.status == 200:
                file_name = "generated_image.png"
                with open(file_name, 'wb') as file:
                    file.write(await response.read())
                st.image(file_name, caption="Generated Image", use_container_width=True)
                st.success("Image generated and displayed successfully!")
            else:
                st.error(f"Failed to download the image. Status Code: {response.status}")
    except asyncio.TimeoutError:
        st.error("The request timed out while fetching the image.")
    except Exception as e:
        st.error(f"An error occurred while fetching the image: {e}")

async def generate_image_and_fetch(text):
    st.write("Generating image... This may take a moment.")
    input_data = {
        'prompt': f'{text}',
        'negprompt': 'deformed, bad anatomy, disfigured, poorly drawn face',
        'samples': 1,
        'steps': 30,
        'aspect_ratio': 'square',
        'guidance_scale': 7.0,  
        'seed': 2414,
    }
    try:
        # Generate image with MonsterAPI asynchronously
        result = monster_client.generate('txt2img', input_data, timeout=60)
        img_url = result['output'][0]

        async with aiohttp.ClientSession() as session:
            await fetch_image_async(session, img_url)
    except Exception as e:
        st.error(f"An error occurred while generating the image: {e}")

if st.button("Click to Record Audio"):
    st.write("Recording... Please speak.")
    
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source,timeout=None, phrase_time_limit=20)

            try:
                text = recognizer.recognize_google(audio, language="en-US")
                st.success(f"Recognized Speech: {text}")
                asyncio.run(generate_image_and_fetch(text))
            except sr.UnknownValueError:
                st.warning("Sorry, couldn't understand the audio.")
            except sr.RequestError:
                st.error("Error with Google Speech Recognition API.")
    except Exception as general_error:
        st.error(f"Unexpected error: {general_error}")

# Adding an audio file upload option
uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg", "flac", "m4a"])

if uploaded_file:
    try:
        audio_bytes = uploaded_file.read()
        audio_data = AudioSegment.from_file(BytesIO(audio_bytes), format=uploaded_file.name.split('.')[-1])
        audio_buffer = BytesIO()
        audio_data.export(audio_buffer, format="wav")
        audio_buffer.seek(0)

        with sr.AudioFile(audio_buffer) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="en-US")
                st.success(f"Recognized Speech from Uploaded File: {text}")
                asyncio.run(generate_image_and_fetch(text))
            except sr.UnknownValueError:
                st.warning("Sorry, couldn't understand the audio in the uploaded file.")
            except sr.RequestError:
                st.error("Error with Google Speech Recognition API while processing the uploaded file.")
    except Exception as e:
        st.error(f"Error processing the uploaded audio file: {e}")
