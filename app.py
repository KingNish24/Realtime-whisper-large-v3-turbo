import spaces
import torch
import gradio as gr
from transformers import pipeline
import tempfile
import os
import uuid
import scipy.io.wavfile
import numpy as np

MODEL_NAME = "ylacombe/whisper-large-v3-turbo"
BATCH_SIZE = 8
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

@spaces.GPU
def transcribe(inputs, previous_transcription):
    try:
        sample_rate, audio_data = inputs

        # Convert audio data to a NumPy array of floats normalized between -1 and 1
        audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Perform transcription
        transcription = pipe(audio_data, 
                             batch_size=BATCH_SIZE, 
                             generate_kwargs={"task": "transcribe"}, 
                             return_timestamps=True)

        # Append new transcription to previous transcription
        previous_transcription += transcription["text"]

        return previous_transcription
    except Exception as e:
        print(f"Error during transcription: {e}")
        return previous_transcription  

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(f"# Realtime Whisper Large V3 Turbo: Transcribe Audio\n Transcribe inputs in Realtime. This Demo uses the checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers.")
        input_audio_microphone = gr.Audio(streaming=True)
        output = gr.Textbox(label="Transcription", value="")

        input_audio_microphone.stream(transcribe, [input_audio_microphone, output], [output], time_limit=45, stream_every=2, concurrency_limit=None)

demo.queue().launch()