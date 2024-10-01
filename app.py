import spaces
import torch
import gradio as gr
import tempfile
import os
import uuid
import scipy.io.wavfile
import time  
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
MODEL_NAME = "ylacombe/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_NAME, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="en")

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=25,
    torch_dtype=torch_dtype,
    device=device,
)

@spaces.GPU
def transcribe(inputs, previous_transcription):
    start_time = time.time() 
    try:
        filename = f"{uuid.uuid4().hex}.wav"
        sample_rate, audio_data = inputs
        scipy.io.wavfile.write(filename, sample_rate, audio_data)

        transcription = pipe(filename)["text"]
        previous_transcription += transcription

        end_time = time.time()
        latency = end_time - start_time
        return previous_transcription, str(latency)
    except Exception as e:
        print(f"Error during Transcription: {e}")
        return previous_transcription, "Error"


def clear():
    return ""

with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(f"# Realtime Whisper Large V3 Turbo: \n Transcribe Audio in Realtime. This Demo uses the Checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers.\n Note: The first token takes about 5 seconds. After that, it works flawlessly.")
        with gr.Row():
            input_audio_microphone = gr.Audio(streaming=True)
            output = gr.Textbox(label="Transcription", value="")
            latency_textbox = gr.Textbox(label="Latency (seconds)", value="0.0", scale=0)
        with gr.Row():
            clear_button = gr.Button("Clear Output")

        input_audio_microphone.stream(transcribe, [input_audio_microphone, output], [output, latency_textbox], time_limit=45, stream_every=2, concurrency_limit=None)
        clear_button.click(clear, outputs=[output])

demo.launch()