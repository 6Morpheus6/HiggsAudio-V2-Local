import gradio as gr
import torch
import torchaudio
import tempfile
import os
import numpy as np
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# Initialize the model
device = "cuda" if torch.cuda.is_available() else "cpu"
serve_engine = None

def initialize_model():
    global serve_engine
    if serve_engine is None:
        try:
            # Load the model using the proper serve engine
            serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

            return f"Model loaded on {device}"
        except Exception as e:
            return f"Error loading model: {str(e)}"

def generate_speech(text, scene_description="Audio is recorded from a quiet room.", temperature=0.3, top_p=0.95, top_k=50, max_tokens=1024):
    if not text.strip():
        return None, "Please enter some text to generate speech."

    try:
        # Initialize model if not already done
        if serve_engine is None:
            init_result = initialize_model()
            if "Error" in init_result:
                return None, init_result

        # Create the system prompt following Higgs Audio format
        system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"

        # Create messages in the proper format
        messages = [
            Message(
                role="system",
                content=system_prompt,
            ),
            Message(
                role="user",
                content=text,
            ),
        ]

        # Generate audio using the serve engine
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        # Save the generated audio to a temporary file
        if output.audio is not None and len(output.audio) > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                torchaudio.save(tmp_file.name, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                return tmp_file.name, f"Speech generated successfully! Sample rate: {output.sampling_rate}Hz"
        else:
            return None, "No audio was generated. The model may have produced only text output."

    except Exception as e:
        return None, f"Error generating speech: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Higgs Audio V2 Text-to-Speech") as demo:
    gr.Markdown("# Higgs Audio V2 Text-to-Speech Interface")
    gr.Markdown("Generate expressive speech from text using Higgs Audio V2 model.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to speak",
                placeholder="Enter the text you want to convert to speech...",
                lines=3
            )
            
            scene_input = gr.Textbox(
                label="Scene Description",
                value="Audio is recorded from a quiet room.",
                placeholder="Describe the audio scene/environment...",
                lines=2
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="Temperature (creativity)"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="Top-p (nucleus sampling)"
                )
                
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k (token selection)"
                )
                
                max_tokens = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=1024,
                    step=256,
                    label="Max tokens"
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
            status_output = gr.Textbox(label="Status", interactive=False)
    
    # Model initialization section
    with gr.Row():
        init_btn = gr.Button("Initialize Model")
        init_status = gr.Textbox(label="Model Status", interactive=False)
    
    # Event handlers
    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, scene_input, temperature, top_p, top_k, max_tokens],
        outputs=[audio_output, status_output]
    )
    
    init_btn.click(
        fn=initialize_model,
        outputs=init_status
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."],
            ["Hello everyone! Welcome to our presentation about artificial intelligence and its applications."],
            ["Once upon a time, in a distant galaxy, there lived a brave space explorer who discovered new worlds."],
            ["Good morning! The weather today is sunny with a gentle breeze. Perfect for a walk in the park."]
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)