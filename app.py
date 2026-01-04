import gradio as gr
import torch
from TTS.api import TTS
import asyncio
import edge_tts
import os
import librosa
import soundfile as sf
import numpy as np

# डिवाइस सेटअप (GPU है तो बहुत तेज़ चलेगा)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# XTTS v2 मॉडल लोड करना
print('📥 Loading Realistic XTTS v2 Model...')
try:
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
except Exception as e:
    print(f"Model loading error: {e}")

# ऑडियो क्लीनअप और साइलेंस रिमूवर फंक्शन
def cleanup_audio(audio_path, remove_silence=True):
    y, sr = librosa.load(audio_path)
    
    if remove_silence:
        # शांत हिस्सों को हटाना (Silence Remover)
        y, _ = librosa.effects.trim(y, top_db=20)
    
    # नॉर्मलाइज़ेशन (आवाज़ को साफ़ और बैलेंस्ड करना)
    y = librosa.util.normalize(y)
    
    clean_path = "cleaned_output.wav"
    sf.write(clean_path, y, sr)
    return clean_path

# Standard TTS (Edge-TTS) - तेज़ जनरेशन के लिए
async def fast_tts(text, voice, speed, pitch):
    output = 'fast_voice.mp3'
    # Speed और Pitch को फॉर्मेट करना (+10% या -10%)
    rate = f"{speed:+}%"
    p = f"{pitch:+}Hz"
    
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=p)
    await communicate.save(output)
    return output

# Voice Cloning (XTTS) - असली जैसी आवाज़ के लिए
def clone_voice(text, audio_sample, cleanup):
    output_path = 'cloned_voice.wav'
    # क्लोनिंग प्रोसेस
    tts.tts_to_file(text=text, speaker_wav=audio_sample, language='hi', file_path=output_path)
    
    # अगर यूजर ने क्लीनअप चुना है
    if cleanup:
        output_path = cleanup_audio(output_path)
        
    return output_path

# UI (Gradio) इंटरफेस
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown('# 🎙️ VoiceBatch Studio v2.0.1 - Realistic Edition')
    gr.Markdown('### *High-Pitch, Low-Speed, Silence Remover & Realistic Cloning*')
    
    with gr.Tabs():
        # टैब 1: प्रोफेशनल क्लोनिंग
        with gr.TabItem('🧬 Voice Cloning (Premium)'):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label='यहाँ टेक्स्ट लिखें (Hindi/English)', lines=5)
                    sample = gr.Audio(label='वॉइस सैंपल अपलोड करें (5-10 सेकंड)', type='filepath')
                    use_cleanup = gr.Checkbox(label="Audio Cleanup & Silence Remover", value=True)
                    btn_clone = gr.Button('Clone & Enhance 🚀', variant='primary')
                output_clone = gr.Audio(label='Realistic क्लोन किया हुआ ऑडियो')
            
            btn_clone.click(clone_voice, [input_text, sample, use_cleanup], output_clone)
            
        # टैब 2: तेज़ जनरेशन (Edge-TTS)
        with gr.TabItem('⚡ Ultra-Fast Generation'):
            with gr.Row():
                with gr.Column():
                    t_text = gr.Textbox(label='टेक्स्ट लिखें', lines=5)
                    v_drop = gr.Dropdown(choices=['hi-IN-MadhurNeural', 'hi-IN-SwaraNeural', 'en-US-GuyNeural'], label='आवाज़ चुनें', value='hi-IN-MadhurNeural')
                    with gr.Row():
                        spd_slider = gr.Slider(minimum=-50, maximum=50, value=0, label="Speed (%)")
                        ptc_slider = gr.Slider(minimum=-20, maximum=20, value=0, label="Pitch (Hz)")
                    btn_fast = gr.Button('Generate Fast ⚡')
                output_fast = gr.Audio(label='साफ़ और तेज़ ऑडियो')
            
            btn_fast.click(lambda t, v, s, p: asyncio.run(fast_tts(t, v, s, p)), [t_text, v_drop, spd_slider, ptc_slider], output_fast)

    gr.Markdown('---')
    gr.Markdown('**नोट:** पहली बार क्लोनिंग करने में मॉडल डाउनलोड होने के कारण समय लग सकता है।')

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
