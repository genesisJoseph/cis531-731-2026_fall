import sys
import hashlib

def verify_environment():
    try:
        import torch
        import transformers
        import mlflow
        import faiss
        import whisper
    except ImportError as e:
        print(f"FAILED: Missing module - {e}")
        return

    # Extract versions
    env_str = f"torch-{torch.__version__}_transformers-{transformers.__version__}_mlflow-{mlflow.__version__}_faiss-{faiss.__version__}"
    
    # Generate SHA-256 Digest
    digest = hashlib.sha256(env_str.encode('utf-8')).hexdigest()
    print(f"Environment String: {env_str}")
    print(f"SHA-256 Digest: {digest}")

if __name__ == "__main__":
    verify_environment()