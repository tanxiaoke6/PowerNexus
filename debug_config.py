
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from config.settings import config
try:
    import openai
    print("OpenAI module found version:", openai.__version__)
    OPENAI_FOUND = True
except ImportError:
    print("OpenAI module NOT found")
    OPENAI_FOUND = False

print(f"Config QwenVL API Key: {config.qwen_vl.api_key}")
print(f"Config QwenLLM API Key: {config.qwen_llm.api_key}")

if OPENAI_FOUND:
    print("API mode should be enabled.")
else:
    print("API mode will be disabled due to missing openai module.")
