"""Publish hf/ (checkpoint, model card, eval files, inference code) to the Hub.
Needs a write login first: `hf auth login`. usage: python scripts/hf_upload.py [repo_id]"""
import sys
from huggingface_hub import HfApi

repo = sys.argv[1] if len(sys.argv) > 1 else "a3ro-dev/reflex-2048"
api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, folder_path="hf", commit_message="reflex-2048 v0.1: 164k mean, no search, 14.5M params",
                  ignore_patterns=["**/__pycache__/**"])
print(f"https://huggingface.co/{repo}")
