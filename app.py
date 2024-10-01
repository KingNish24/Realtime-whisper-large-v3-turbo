import spaces
import torch
import gradio as gr
from transformers import pipeline
import tempfile
import os
import uuid
import scipy.io.wavfile

MODEL_NAME = "ylacombe/whisper-large-v3-turbo"
BATCH_SIZE = 16  
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
        # Generate a unique filename Using UUID
        filename = f"{uuid.uuid4().hex}.wav"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        # Extract Sample Rate and Audio Data from the Tuple
        sample_rate, audio_data = inputs

        # Save the Audio Data to the Temporary File
        scipy.io.wavfile.write(filepath, sample_rate, audio_data)

        # Transcribe the Audio
        transcription = pipe(filepath, batch_size=BATCH_SIZE, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
        previous_transcription += transcription

        # Remove the Temporary File after Transcription
        os.remove(filepath)

        return previous_transcription
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return previous Transcription

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(f"# Realtime Whisper Large V3 Turbo: Transcribe Audio\n Transcribe Inputs in Realtime. This Demo uses the Checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers.")
        input_audio_microphone = gr.Audio(streaming=True)
        output = gr.Textbox(label="Transcription", value="")

        input_audio_microphone.stream(transcribe, [input_audio_microphone, output], [output], time_limit=45, stream_every=2, concurrency_limit=None)

demo.launch()