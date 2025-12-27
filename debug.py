import sys
import os

print(f"Python Executable: {sys.executable}")
try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
    print(f"Transformers Location: {transformers.__file__}")
except ImportError as e:
    print(f"CRITICAL: Could not import transformers! {e}")

print("-" * 20)
print("Attempting to import PaliGemma...")
try:
    from transformers import PaliGemmaForConditionalGeneration
    print("✅ SUCCESS! PaliGemmaForConditionalGeneration imported.")
except ImportError as e:
    print("❌ FAILED to import PaliGemma.")
    print(f"The REAL error is: {e}")

    # Check for common missing sub-dependencies
    try:
        import sentencepiece
        print("Sentencepiece is installed.")
    except ImportError:
        print("⚠️ Sentencepiece is MISSING.")

    try:
        import google.protobuf
        print(f"Protobuf is installed: {google.protobuf.__version__}")
    except ImportError:
        print("⚠️ Protobuf is MISSING or BROKEN.")