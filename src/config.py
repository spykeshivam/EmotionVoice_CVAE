"""Configuration parameters for audio emotion transfer project."""

import os
import torch

# Paths
AUDIO_NEUTRAL_DIR = "./Audio_transformed"  # Contains mp3 files without emotion
AUDIO_EMOTION_DIR = "./datasets"  # Contains wav files with emotions
OUTPUT_DIR = "./outputs"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# Create output directory if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Audio parameters
SAMPLE_RATE = 22050
MAX_AUDIO_LENGTH = SAMPLE_RATE * 5  # 5 seconds of audio
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 80
GL_ITERS = 60

# Model parameters
LATENT_DIM = 128
EMOTION_DIM = 2  # Angry or Happy
EMOTION_CATEGORIES = ['Angry_padded', 'Happy_padded']

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
BETA = 1.0  # Weight for KL divergence term in loss

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seeds for reproducibility
RANDOM_SEED = 42