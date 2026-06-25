#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local model training launcher.

This project integrates three local algorithm modules:

1. Transformer classifier for text sentiment recognition.
2. CNN-based pipeline for facial expression recognition.
3. Neural speech interaction module for speech input/output.

The script prints the exact commands to run because each algorithm module has
its own training entry points and GPU requirements.
"""

from pathlib import Path


GROUP_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = GROUP_ROOT / "third_party"


def main():
    sentiment = THIRD_PARTY_ROOT / "text_sentiment_recognition"
    paz = THIRD_PARTY_ROOT / "facial_expression_recognition"
    mini = THIRD_PARTY_ROOT / "speech_interaction"
    config = GROUP_ROOT / "data" / "model_config.json"

    print("=" * 70)
    print("Local model training / replacement guide")
    print("=" * 70)
    print()
    print("Text sentiment model:")
    print(f"  cd \"{sentiment}\"")
    print("  python train.py --model_name_or_path transformer-base --output_dir my_model --num_eps 2")
    print("  Then set text_sentiment_model_path to the trained model directory.")
    print()
    print("Facial expression model:")
    print(f"  cd \"{paz}\"")
    print("  python examples/face_classification/train.py")
    print("  Then keep facial_expression_project pointing to this module, or replace its FER weights.")
    print()
    print("Voice model:")
    print(f"  cd \"{mini}\"")
    print("  Follow the speech module training/fine-tuning instructions, then set speech_generation_checkpoint")
    print("  to the checkpoint directory containing lit_model.pth and model_config.yaml.")
    print()
    print("Emotional recovery dialogue model (optional):")
    print("  Train or fine-tune a Hugging Face compatible causal language model on")
    print("  emotional-support conversations, then set local_dialogue_model_path to")
    print("  the saved model directory. The app will use it before the lightweight fallback.")
    print()
    print(f"Current app model config file: {config}")
    print("=" * 70)


if __name__ == "__main__":
    main()
