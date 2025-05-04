"""Visualization functions for audio and spectrograms."""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import librosa
import librosa.display

def plot_spectrograms(neutral_spec, recon_batch, emotion_label, epoch, idx):
    """
    Plot original neutral spectrogram and reconstructed spectrogram side by side
    
    Args:
        neutral_spec: Original neutral spectrogram tensor
        recon_batch: Reconstructed spectrogram tensor
        emotion_label: Emotion label used for conditioning
        epoch: Current training epoch
        idx: Sample index
    """
    
    
    
    
    # Convert tensors to numpy arrays
    neutral_spec_np = np.squeeze(neutral_spec.cpu().detach().numpy())  
    recon_spec_np = np.squeeze(recon_batch.cpu().detach().numpy())     
    print(recon_spec_np.shape)

    
    # Get emotion name from label
    emotions = ["Angry", "Angry"]
    
    # Handle different possible formats of emotion_label
    try:
        if emotion_label.dim() > 0 and emotion_label.size(0) == 2:
            emotion_idx = emotion_label.argmax().item()
            emotion_name = emotions[emotion_idx]
        else:
            # Index case (scalar tensor)
            emotion_name = emotions[emotion_label.item()]
    except (IndexError, ValueError, AttributeError, RuntimeError):
        # Fallback - just display the tensor content
        emotion_name = f"Emotion {emotion_label}"
    
    # Create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original spectrogram
    im0 = axs[0].imshow(neutral_spec_np, origin='lower', aspect='auto', cmap='viridis')
    axs[0].set_title("Original Neutral Spectrogram")
    axs[0].set_ylabel("Frequency Bins")
    axs[0].set_xlabel("Time Steps")
    
    # Plot reconstructed spectrogram
    im1 = axs[1].imshow(recon_spec_np, origin='lower', aspect='auto', cmap='viridis')
    axs[1].set_title(f"Reconstructed Spectrogram\nTarget Emotion: {emotion_name}")
    axs[1].set_xlabel("Time Steps")
    
    # Add colorbar
    plt.colorbar(im0, ax=axs[0], label="Magnitude")
    plt.colorbar(im1, ax=axs[1], label="Magnitude")
    
    # Add superTitle
    plt.suptitle(f"Epoch {epoch}, Sample {idx}")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.show()
    plt.close()
