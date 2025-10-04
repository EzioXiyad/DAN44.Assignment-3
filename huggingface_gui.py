"""
Hugging Face Models Integration with Gradio
-------------------------------------------
This script integrates two Hugging Face models:
1. Image Classification (Vision Transformer)
2. Speech Recognition (Whisper)

It provides a Gradio GUI where users can:
- Upload an image and classify it.
- Upload an audio clip and transcribe it.
"""

import gradio as gr
from transformers import pipeline

# ---------------------------
# Load Hugging Face Pipelines
# ---------------------------

# Image Classification Model (Vision Transformer)
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Automatic Speech Recognition Model (Whisper)
asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-base")


# ---------------------------
# Wrapper Functions
# ---------------------------

def classify_image(img):
    """
    Classifies an input image using the Vision Transformer.
    Returns top predictions with labels and scores.
    """
    results = image_classifier(img)
    return results


def transcribe_audio(audio_file):
    """
    Transcribes an input audio file using Whisper.
    Returns the transcribed text.
    """
    results = asr_model(audio_file)
    return results["text"]


# ---------------------------
# Build Gradio GUI
# ---------------------------

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Hugging Face Model Integration Demo")

    with gr.Tab("📷 Image Classification"):
        gr.Markdown("Upload an image and classify it using ViT model")
        img_input = gr.Image(type="filepath", label="Upload Image")
        img_output = gr.JSON(label="Predictions")
        img_button = gr.Button("Classify")
        img_button.click(fn=classify_image, inputs=img_input, outputs=img_output)

    with gr.Tab("🎙️ Speech Recognition"):
        gr.Markdown("Upload an audio file and transcribe it using Whisper model")
        audio_input = gr.Audio(type="filepath", label="Upload Audio")
        audio_output = gr.Textbox(label="Transcription")
        audio_button = gr.Button("Transcribe")
        audio_button.click(fn=transcribe_audio, inputs=audio_input, outputs=audio_output)


# ---------------------------
# Launch App
# ---------------------------

if __name__ == "__main__":
    demo.launch()
