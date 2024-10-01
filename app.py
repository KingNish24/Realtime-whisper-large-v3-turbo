import spaces
import torch
import gradio as gr
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile
import os

MODEL_NAME = "ylacombe/whisper-large-v3-turbo"
BATCH_SIZE = 8
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=1,
    device=device,
)

@spaces.GPU
def transcribe(inputs, previous_transcription):
    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    if previous_transcription:
        text = previous_transcription + text
    return text

with gr.Blocks() as demo:
     with gr.Column():
        input_audio_microphone = gr.Audio(streaming=True),
        output = gr.Textbox("Transcription")

    input_audio_microphone.stream(
        transcribe, 
        [input_audio, output],
        [output],
        time_limit=15,
        stream_every=0.5,
        concurrency_limit=None
    )

demo.queue().launch()
