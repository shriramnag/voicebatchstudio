import os
import sys

# फोल्डर पाथ सेट करना (ताकि कोई एरर न आए)
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'src'))

# अब यहाँ अपना Chatterbox वाला कोड पेस्ट करें...

import os
import torch
import gradio as gr
import edge_tts
import asyncio
import tempfile
import numpy as np
import librosa
from pathlib import Path

# आपके द्वारा बनाए गए मॉड्यूल्स को इम्पोर्ट करना
from voicebatchstudio.tts import ChatterboxTTS
from voicebatchstudio.vc import ChatterboxVC
from voicebatchstudio.mtl_tts import ChatterboxMTL

# --- कॉन्फ़िगरेशन ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# मॉडल लोड करना
print(f"Loading Models on {DEVICE}...")
tts_engine = ChatterboxTTS.from_pretrained(device=DEVICE)
vc_engine = ChatterboxVC.from_pretrained(device=DEVICE)

# --- फ्री मॉडल्स (TTS 5) फ़ंक्शंस ---

async def generate_edge_tts(text, voice="hi-IN-MadhurNeural"):
    """Microsoft Edge TTS (Free & High Quality Hindi)"""
    communicate = edge_tts.Communicate(text, voice)
    path = os.path.join(OUTPUT_DIR, "edge_out.mp3")
    await communicate.save(path)
    return path

def clone_voice(text, ref_audio, speed):
    """Chatterbox Turbo Cloning"""
    if ref_audio is None:
        return None
    # ऑडियो जनरेट करें
    wav = tts_engine.generate(text, ref_audio, speed=speed)
    path = os.path.join(OUTPUT_DIR, "clone_out.wav")
    import soundfile as sf
    sf.write(path, wav.squeeze().cpu().numpy(), 24000)
    return path

def convert_voice(source_audio, target_audio):
    """Voice Conversion (आवाज़ बदलें)"""
    if source_audio is None or target_audio is None:
        return None
    wav = vc_engine.generate(source_audio, target_voice_path=target_audio)
    path = os.path.join(OUTPUT_DIR, "vc_out.wav")
    import soundfile as sf
    sf.write(path, wav.squeeze().cpu().numpy(), 24000)
    return path

# --- Gradio UI डिज़ाइन ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ VoiceBatch Studio (Free AI Voice)")
    gr.Markdown("### सब कुछ फ्री, हमेशा के लिए। (हिंदी और इंग्लिश सपोर्ट)")

    with gr.Tabs():
        # टैब 1: सुपरफास्ट क्लोनिंग
        with gr.TabItem("🚀 Turbo Cloning"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(label="टेक्स्ट लिखें", placeholder="यहाँ कुछ लिखें...", lines=3)
                    ref_audio = gr.Audio(label="अपनी आवाज़ अपलोड करें (5-10 सेकंड)", type="filepath")
                    speed = gr.Slider(0.5, 2.0, value=1.0, label="बोलने की रफ़्तार (Speed)")
                    btn_tts = gr.Button("Generate Clone", variant="primary")
                with gr.Column():
                    audio_output = gr.Audio(label="AI की आवाज़")

            btn_tts.click(clone_voice, inputs=[text_input, ref_audio, speed], outputs=audio_output)

        # टैब 2: फ्री हिंदी आवाज़ें (TTS 5)
        with gr.TabItem("🇮🇳 Free Hindi (Edge-TTS)"):
            with gr.Row():
                with gr.Column():
                    hi_text = gr.Textbox(label="हिंदी टेक्स्ट", value="नमस्ते, आप कैसे हैं?")
                    hi_voice = gr.Dropdown(
                        choices=["hi-IN-MadhurNeural", "hi-IN-SwaraNeural", "en-IN-PrabhatNeural"], 
                        value="hi-IN-MadhurNeural", 
                        label="आवाज़ चुनें"
                    )
                    btn_edge = gr.Button("Generate Hindi Audio")
                with gr.Column():
                    hi_output = gr.Audio(label="Output")
            
            btn_edge.click(lambda t, v: asyncio.run(generate_edge_tts(t, v)), inputs=[hi_text, hi_voice], outputs=hi_output)

        # टैब 3: आवाज़ बदलें (Voice Changer)
        with gr.TabItem("🔄 Voice Changer"):
            with gr.Row():
                with gr.Column():
                    src_aud = gr.Audio(label="जिसकी आवाज़ बदलनी है (File/Record)", type="filepath")
                    tgt_aud = gr.Audio(label="जिसके जैसा बनाना है (Target)", type="filepath")
                    btn_vc = gr.Button("Convert Voice")
                with gr.Column():
                    vc_output = gr.Audio(label="Converted Audio")
            
            btn_vc.click(convert_voice, inputs=[src_aud, tgt_aud], outputs=vc_output)

    gr.Markdown("---")
    gr.Markdown("Built with ❤️ using Chatterbox & Edge-TTS")

# ऐप लॉन्च करें
if __name__ == "__main__":
    demo.launch(share=True, debug=True)
