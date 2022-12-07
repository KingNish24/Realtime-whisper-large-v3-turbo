import torch

import gradio as gr
import pytube as pt
from transformers import pipeline

MODEL_NAME = "openai/whisper-tiny"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


all_special_ids = pipe.tokenizer.all_special_ids
transcribe_token_id = all_special_ids[-5]
translate_token_id = all_special_ids[-6]


def transcribe(microphone, file_upload, do_translate):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    pipe.model.config.forced_decoder_ids = [[2, translate_token_id if do_translate else transcribe_token_id]]

    text = pipe(file)["text"]

    return warn_output + text


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url, do_translate):
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio.mp3")

    pipe.model.config.forced_decoder_ids = [[2, translate_token_id if do_translate else transcribe_token_id]]

    text = pipe("audio.mp3")["text"]

    return html_embed_str, text


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Audio(source="upload", type="filepath", optional=True),
        gr.Checkbox(label="Translate?", value=False),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the the fine-tuned"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"), gr.Checkbox(label="Translate?", value=False)],
    outputs=["html", "text"],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Demo: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the checkpoint:"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files of"
        " arbitrary length."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Audio", "Transcribe YouTube"])

demo.launch(enable_queue=True, share=True)
