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

# Preset configurations
PRESETS = {
    "default": {
        "scene_description": "Audio is recorded from a quiet room.",
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 1024
    },
    "female_voice": {
        "scene_description": "Audio is recorded from a quiet room with a clear female voice.",
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 1024
    },
    "male_voice": {
        "scene_description": "Audio is recorded from a quiet room with a clear male voice.",
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 1024
    },
    "high_quality": {
        "scene_description": "Audio is recorded in a professional studio with high-quality equipment.",
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 30,
        "max_tokens": 1024
    },
    "creative": {
        "scene_description": "Audio is recorded with natural expression and creativity.",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 80,
        "max_tokens": 1024
    },
    "fast": {
        "scene_description": "Audio is recorded from a quiet room.",
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 512
    }
}

# Default settings (for backward compatibility)
DEFAULT_SETTINGS = PRESETS["default"]

def load_preset_settings(preset_name="default"):
    """Load preset settings for all parameters"""
    preset = PRESETS.get(preset_name, PRESETS["default"])
    return (
        preset["scene_description"],
        preset["temperature"],
        preset["top_p"],
        preset["top_k"],
        preset["max_tokens"],
        f"'{preset_name.title()}' preset loaded successfully!"
    )

def load_default_settings():
    """Load default settings for all parameters"""
    return load_preset_settings("default")

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
                value=DEFAULT_SETTINGS["scene_description"],
                placeholder="Describe the audio scene/environment...",
                lines=2
            )

            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=DEFAULT_SETTINGS["temperature"],
                    step=0.1,
                    label="Temperature (creativity)"
                )

                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=DEFAULT_SETTINGS["top_p"],
                    step=0.05,
                    label="Top-p (nucleus sampling)"
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=DEFAULT_SETTINGS["top_k"],
                    step=1,
                    label="Top-k (token selection)"
                )

                max_tokens = gr.Slider(
                    minimum=256,
                    maximum=2048,
                    value=DEFAULT_SETTINGS["max_tokens"],
                    step=256,
                    label="Max tokens"
                )

                # Preset settings
                with gr.Row():
                    preset_dropdown = gr.Dropdown(
                        choices=[
                            ("Default - Balanced quality and speed", "default"),
                            ("Female Voice - Optimized for female speech", "female_voice"),
                            ("Male Voice - Optimized for male speech", "male_voice"),
                            ("High Quality - Best quality, conservative settings", "high_quality"),
                            ("Creative - More expressive and varied output", "creative"),
                            ("Fast - Quick generation with shorter output", "fast")
                        ],
                        value="default",
                        label="Presets",
                        info="Choose a preset configuration"
                    )
                    load_preset_btn = gr.Button("📋 Load Preset", variant="secondary")

                defaults_btn = gr.Button("🔄 Reset to Default", variant="secondary")

            generate_btn = gr.Button("Generate Speech", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
            status_output = gr.Textbox(label="Status", interactive=False)
            defaults_status = gr.Textbox(label="Settings Status", interactive=False)

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

    defaults_btn.click(
        fn=load_default_settings,
        outputs=[scene_input, temperature, top_p, top_k, max_tokens, defaults_status]
    )

    load_preset_btn.click(
        fn=load_preset_settings,
        inputs=[preset_dropdown],
        outputs=[scene_input, temperature, top_p, top_k, max_tokens, defaults_status]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years."],
            ["Hello everyone! Welcome to our presentation about artificial intelligence and its applications."],
            ["Once upon a time, in a distant galaxy, there lived a brave space explorer who discovered new worlds."],
            ["Good morning! The weather today is sunny with a gentle breeze. Perfect for a walk in the park."],
            ["[FEMALE] Thank you for joining us today. I'm excited to share these important findings with you."],
            ["[MALE] Welcome to the conference. Let's begin with our first presentation."]
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Higgs Audio V2 Gradio Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Enable public sharing via Gradio")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Disable public sharing (default)")
    parser.set_defaults(share=False)

    args = parser.parse_args()

    print(f"Starting Higgs Audio V2 interface on {args.host}:{args.port}")
    if args.share:
        print("Public sharing enabled via Gradio")
    else:
        print("Local access only")

    demo.launch(
        share=args.share,
        server_name=args.host,
        server_port=args.port
    )