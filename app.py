import spaces
import torch
import gradio as gr
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import tempfile
import os

MODEL_NAME = "ylacombe/whisper-large-v3-turbo"
BATCH_SIZE = 32
device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=1,
    device=device,
)

@spaces.GPU
def transcribe(inputs, previous_transcription):
    previous_transcription += pipe(inputs[1], batch_size=BATCH_SIZE, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
    return previous_transcription

with gr.Blocks() as demo:
    with gr.Column():
        input_audio_microphone = gr.Audio(streaming=True)
        output = gr.Textbox(label="Transcription", value="")
        
        input_audio_microphone.stream(transcribe, [input_audio_microphone, output], [output], time_limit=15, stream_every=1, concurrency_limit=None)

demo.queue().launch()