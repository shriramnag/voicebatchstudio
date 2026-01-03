import torch

# यह डिक्शनरी आपके 'ImportError' को हमेशा के लिए खत्म कर देगी
COSYVOICE_EMB_CLASSES = {
    'speech_tokenizer': 'voicebatchstudio.models.s3tokenizer.s3tokenizer.S3Gen',
    'embed': torch.nn.Embedding,
}

COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "swish": torch.nn.SiLU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    'paraformer_dummy': torch.nn.Identity
}

COSYVOICE_ATTENTION_CLASSES = {}
