try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

# चूंकि अभी हमने इसे पैकेज की तरह इंस्टॉल नहीं किया है, 
# इसलिए हम वर्जन को मैन्युअली 0.1.0 रख सकते हैं
__version__ = "0.1.0"

# यहाँ हम 'voicebatchstudio' के अंदर की फाइलों को इम्पोर्ट कर रहे हैं
from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
