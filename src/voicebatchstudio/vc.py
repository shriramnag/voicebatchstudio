from pathlib import Path
import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# --- voicebatchstudio के हिसाब से सही पाथ ---
from voicebatchstudio.models.s3tokenizer import S3_SR
from voicebatchstudio.models.s3gen import S3GEN_SR, S3Gen

REPO_ID = "ResembleAI/chatterbox"

class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, s3gen, device, ref_dict=None):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    # ... (बाकी का from_local, from_pretrained और generate कोड वैसा ही रहेगा जैसा आपने दिया है) ...
