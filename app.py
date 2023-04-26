import torch

import gradio as gr
import pytube as pt
from transformers import pipeline

MODEL_NAME = "openai/whisper-large-v2"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000
YT_ATTEMPT_LIMIT = 3

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


def transcribe(microphone, file_upload, task):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        raise gr.Error("You have to either use the microphone or upload an audio file")

    file_size_mb = os.stat(inputs).st_size / (1024 * 1024)
    if file_size_mb > FILE_LIMIT_MB:
        raise gr.Error(
                f"File size exceeds file size limit. Got file of size {file_size_mb:.2f}MB for a limit of {FILE_LIMIT_MB}MB."
        )

    file = microphone if microphone is not None else file_upload

    pipe.model.config.forced_decoder_ids = [[2, transcribe_token_id if task=="transcribe" else translate_token_id]]

    text = pipe(file, batch_size=BATCH_SIZE)["text"]

    return warn_output + text


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url, task, max_filesize=75.0):
    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)
    for attempt in range(YT_ATTEMPT_LIMIT):
        try:
            yt = pytube.YouTube(yt_url)
            stream = yt.streams.filter(only_audio=True)[0]
            break
        except KeyError:
            if attempt + 1 == YT_ATTEMPT_LIMIT:
                raise gr.Error("An error occurred while loading the YouTube video. Please try again.")

    if stream.filesize_mb > max_filesize:
        raise gr.Error(f"Maximum YouTube file size is {max_filesize}MB, got {stream.filesize_mb:.2f}MB.")

    pipe.model.config.forced_decoder_ids = [[2, transcribe_token_id if task=="transcribe" else translate_token_id]]

    text = pipe("audio.mp3", batch_size=BATCH_SIZE)["text"]

    return html_embed_str, text


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Audio(source="upload", type="filepath", optional=True),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V2: Transcribe Audio",
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the"
        f" checkpoint [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe audio files"
        " of arbitrary length."
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[
        gr.inputs.Textbox(lines=1, placeholder="Paste the URL to a YouTube video here", label="YouTube URL"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe")
    ],
    outputs=["html", "text"],
    layout="horizontal",
    theme="huggingface",
    title="Whisper Large V2: Transcribe YouTube",
    description=(
        "Transcribe long-form YouTube videos with the click of a button! Demo uses the checkpoint"
        f" [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME}) and 🤗 Transformers to transcribe video files of"
        " arbitrary length."
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, yt_transcribe], ["Transcribe Audio", "Transcribe YouTube"])

demo.launch(enable_queue=True)

