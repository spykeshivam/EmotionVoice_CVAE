"""
Inference script for audio emotion transfer using the trained CVAE model.
Takes an MP3 file, applies emotion transfer, and saves the result.
"""

import os
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import librosa
import soundfile as sf
import pygame
from IPython.display import display, Audio
import time
from pydub import AudioSegment
from pydub.effects import speedup

from src.cvae import CVAE
from src.dataset import AudioEmotionDataset

from config import (
    DEVICE, 
    LATENT_DIM, 
    EMOTION_DIM, 
    LEARNING_RATE, 
    EPOCHS, 
    BETA,
    MODEL_DIR, 
    N_MELS,
    MAX_AUDIO_LENGTH,
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    EMOTION_CATEGORIES,
    GL_ITERS
)

WIN_LENGTH = N_FFT

def convert_to_male_voice(input_audio_file, output_audio_file):
    try:
        # Determine file format from extension
        input_format = os.path.splitext(input_audio_file)[1][1:]
        
        # Load the audio file
        sound = AudioSegment.from_file(input_audio_file, format=input_format)
        
        # Lower the pitch by modifying the sound
        slowdown_factor = 0.85  # 15% slower
        male_voice_sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * slowdown_factor)
        })
        
        # Speed up tempo without affecting pitch
        male_voice_sound = speedup(male_voice_sound, 1.25, 150)
        
        # Apply low-pass filter for a more masculine sound
        male_voice_sound = male_voice_sound.low_pass_filter(300)
        
        # Export the modified audio
        output_format = os.path.splitext(output_audio_file)[1][1:]
        male_voice_sound.export(output_audio_file, format=output_format)
        
        return True
    except Exception as e:
        print(f"Error processing {input_audio_file}: {str(e)}")
        return False



def play_audio_jupyter(file_path, use_pygame=False):
    """
    Play an audio file in a Jupyter notebook
    
    Args:
        file_path (str): Path to the audio file
        use_pygame (bool): Whether to use pygame for playback instead of IPython's Audio
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    if use_pygame:
        # Initialize pygame mixer
        pygame.mixer.init()
        try:
            # Load and play the sound
            sound = pygame.mixer.Sound(file_path)
            print(f"Playing: {file_path}")
            print(f"Duration: {sound.get_length():.2f} seconds")
            
            # Play the sound
            sound.play()
            
            # Keep the program running while the sound is playing
            pygame.time.wait(int(sound.get_length() * 1000))
            
        except Exception as e:
            print(f"Error playing audio with pygame: {e}")
        finally:
            # Clean up resources
            pygame.mixer.quit()
    else:
        try:
            print(f"Playing: {file_path}")
            display(Audio(file_path, autoplay=True))
        except Exception as e:
            print(f"Error playing audio with IPython: {e}")



def transfer_emotion(model_path, mel_spec_db, emotion_index, output_dir="./outputs",model=None):
    """
    Apply emotion transfer to an audio file using the trained model
    
    Args:
        model_path: Path to the trained model file
        audio_path: Path to the input audio file
        emotion_index: Index of the target emotion
        output_dir: Directory to save outputs
        
    Returns:
        output_paths: Dict with paths to original-recon, transformed-recon, and (optionally) MP3
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load model ---
    sample_spec = mel_spec_db
    input_dim = (sample_spec.shape[1], sample_spec.shape[2])
    if model is None:        
        model = CVAE(input_dim=input_dim, latent_dim=LATENT_DIM, emotion_dim=EMOTION_DIM)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
    
    # --- Prepare emotion vector ---
    emotion_tensor = torch.zeros(EMOTION_DIM, device=device)
    emotion_tensor[emotion_index] = 1.0
    emotion_tensor = emotion_tensor.unsqueeze(0)  # batch=1
    
    # --- Get original Mel spectrogram ---
    mel_spec_db = mel_spec_db
    mel_spec_db = mel_spec_db.unsqueeze(0).to(device)   # shape: (1, n_mels, T)
    
    # --- Emotion transfer ---
    with torch.no_grad():
        transformed_spec, _, _ = model(mel_spec_db, emotion_tensor)
    transformed_spec = transformed_spec.cpu()            # shape: (1, n_mels, T)
    
    # --- Print shapes ---
    print(f"Original Mel-spec dB shape: {mel_spec_db.shape}")
    print(f"Transformed spec    shape: {transformed_spec.shape}")
    # remove batch dim, pass to your helper
    orig_recon = spectrogram_to_audio(mel_spec_db.squeeze(0))
    trans_recon = spectrogram_to_audio(transformed_spec.squeeze(0))
    
    # --- Normalize to avoid clipping ---
    def normalize(x):
        m = np.max(np.abs(x))
        return x / m * 0.9 if m > 0 else x
    
    orig_recon = normalize(orig_recon)
    trans_recon = normalize(trans_recon)
    
    # --- Save WAVs ---
    orig_path = os.path.join(output_dir, "recon_original.wav")
    trans_path = os.path.join(output_dir, f"recon_{EMOTION_CATEGORIES[emotion_index]}.wav")
    write(orig_path, SAMPLE_RATE, (orig_recon * 32767).astype(np.int16))
    write(trans_path, SAMPLE_RATE, (trans_recon * 32767).astype(np.int16))
    
    print(f"Saved reconstructed original to: {orig_path}")
    print(f"Saved reconstructed transformed to: {trans_path}")
    
    
    
    #return {
    #    "original_recon_wav": orig_path,
    #    "transformed_recon_wav": trans_path,
    #}
    return transformed_spec


mel_basis = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=0.0,
    fmax=SAMPLE_RATE / 2,
)
mel_inv = np.linalg.pinv(mel_basis)

# Define transforms
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    f_min=0.0,
    f_max=SAMPLE_RATE / 2,
)
def audio_to_spectrogram_transfer(model_path='./outputs/models/best_model.pth',audio_path='./Audio_transformed/Angry_padded/03-01-05-01-01-01-08.mp3', output_dir="./recon_check", emotion='Happy_padded',sample_rate=SAMPLE_RATE,model=None):
    waveform, sr = torchaudio.load(audio_path)
    emotion_index = EMOTION_CATEGORIES.index(emotion)
    
        
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    max_length=sr*5
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    elif waveform.shape[1] < max_length:
        padding = max_length - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 2) Compute Mel-spectrogram in dB using torchaudio
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    mel_spec = mel_spectrogram(waveform)
    mel_db = amplitude_to_db(mel_spec)
    
    # Print shapes for debugging
    #print(f"Mel spectrogram shape: {mel_spec.shape}")
    #print(f"Mel db shape: {mel_db.shape}")
    # Print shapes for debugging
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"Mel db shape: {mel_db.shape}")
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    transformed_spec=transfer_emotion(model_path, mel_db, emotion_index, output_dir="./outputs",model=model)
    transformed_spec=torch.tensor(transformed_spec).squeeze(0)
    #print(f"Mel spectrogram shape: {transformed_spec.shape}")
    #print(f"Mel db shape: {transformed_spec.shape}")
    
    
    
    return mel_db,transformed_spec







# Invert mel spectrogram dB (inverse process of scaling)

def inverse_normalize_spectrogram(normalized_mel_db, target_min=-80.0, target_max=0.0):
    """
    Apply inverse normalization to a normalized mel spectrogram.
    
    This function maps normalized values back to a reasonable dB range
    that can be used for audio reconstruction. Since we don't have the
    original statistics, we use approximate values based on typical
    mel spectrogram characteristics.
    
    Args:
        normalized_mel_db: Normalized mel spectrogram (centered around 0, std ≈ 1)
        target_min: Target minimum dB value (default: -80.0)
        target_max: Target maximum dB value (default: 0.0)
    
    Returns:
        Denormalized mel spectrogram in dB scale
    """
    # Get current min and max
    current_min = normalized_mel_db.min()
    current_max = normalized_mel_db.max()
    #print(current_min,current_max)
    # Rescale to target range
    target_range = target_max - target_min
    current_range = current_max - current_min
    #print(current_range)
    
    # Apply linear scaling
    denormalized = (normalized_mel_db - current_min) / current_range * target_range + target_min    
    return denormalized

def db_to_magnitude(mel_db):
    """Convert from dB (log) to linear-scale mel-spectrogram."""
    return np.power(10.0, mel_db * 0.05)

# pseudo-inverse of mel filterbank
mel_basis = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=0.0,
    fmax=SAMPLE_RATE / 2,
)
mel_inv = np.linalg.pinv(mel_basis)



def spectrogram_to_audio(mel_db):
    """
    Reconstruct waveform from a dB-scaled Mel spectrogram.
    
    Args:
        mel_db: torch.Tensor or np.ndarray of shape (n_mels, t_frames), in dB.
        
    Returns:
        waveform: np.ndarray of shape (n_samples,)
    """
    mel_basis = librosa.filters.mel(
    sr=SAMPLE_RATE,
    n_fft=N_FFT,
    n_mels=N_MELS,
    fmin=0.0,
    fmax=SAMPLE_RATE / 2,
    )
    mel_inv = np.linalg.pinv(mel_basis)

    # To numpy and remove any batch dim
    if hasattr(mel_db, "detach"):
        mel_db = mel_db.detach().cpu().numpy()
    if mel_db.ndim == 3:  # (1, n_mels, t)
        mel_db = mel_db[0]

    mel_db = inverse_normalize_spectrogram(mel_db, -100, 50)
    print("Min value:", mel_db.min().item())
    print("Max value:", mel_db.max().item())

    
    # 2) dB to linear mel magnitude
    mel_mag = np.power(10.0, mel_db * 0.05)
    
    # 3) Mel → approximate full STFT magnitude
    
    stft_mag = np.maximum(1e-10, mel_inv.dot(mel_mag))
    
    # 4) Griffin–Lim to recover phase + waveform
    waveform = librosa.griffinlim(
        stft_mag,
        n_iter=GL_ITERS,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        init="random",
        momentum=0.99,
    )
    return waveform





