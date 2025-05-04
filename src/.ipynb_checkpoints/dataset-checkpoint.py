"""Dataset class for audio emotion transfer with caching."""

import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import (
    AUDIO_NEUTRAL_DIR,
    AUDIO_EMOTION_DIR,
    SAMPLE_RATE,
    MAX_AUDIO_LENGTH,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    BATCH_SIZE,
    EMOTION_CATEGORIES,
    RANDOM_SEED
)

class AudioEmotionDataset(Dataset):
    def __init__(self, neutral_dir=AUDIO_NEUTRAL_DIR, emotion_dir=AUDIO_EMOTION_DIR, 
                 max_length=MAX_AUDIO_LENGTH, sample_rate=SAMPLE_RATE, use_cache=True):
        """
        Dataset for audio emotion transfer with caching.
        
        Args:
            neutral_dir: Directory containing neutral audio files organized by emotion folders
            emotion_dir: Directory containing emotional audio files organized by emotion folders
            max_length: Maximum length of audio in samples
            sample_rate: Target sample rate for audio
            use_cache: Whether to cache processed audio spectrograms
        """
        self.neutral_dir = neutral_dir
        self.emotion_dir = emotion_dir
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.use_cache = use_cache
        
        # Caching dictionaries
        self.neutral_cache = {}
        self.emotion_cache = {}
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Get emotion categories
        self.emotion_categories = EMOTION_CATEGORIES
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotion_categories)}
        
        # Collect all file pairs
        self.file_pairs = []
        self._collect_file_pairs()
    
    def _collect_file_pairs(self):
        """Collect pairs of files: neutral and emotional"""
        for emotion in self.emotion_categories:
            neutral_emotion_dir = os.path.join(self.neutral_dir, emotion)
            emotion_dir = os.path.join(self.emotion_dir, emotion)
            cwd = os.getcwd()
            print("Current working directory:", cwd)
            if not os.path.exists(neutral_emotion_dir) or not os.path.exists(emotion_dir):
                print(f"Warning: Directory not found - {neutral_emotion_dir} or {emotion_dir}")
                continue
            
            neutral_files = [f for f in os.listdir(neutral_emotion_dir) if f.endswith('.mp3')]
            emotion_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            
            # For simplicity, pair files by their index in the sorted list
            # In a real scenario, you'd need a better matching strategy
            num_pairs = min(len(neutral_files), len(emotion_files))
            
            for i in range(num_pairs):
                neutral_file = os.path.join(neutral_emotion_dir, neutral_files[i])
                emotion_file = os.path.join(emotion_dir, emotion_files[i])
                
                self.file_pairs.append({
                    'neutral': neutral_file,
                    'emotion': emotion_file,
                    'emotion_label': self.emotion_to_idx[emotion]
                })
        
        print(f"Collected {len(self.file_pairs)} file pairs for training")
    
    def _load_and_preprocess_audio(self, file_path, is_mp3=False):
        """Load audio file and convert to mel spectrogram"""
        # Check cache first if caching is enabled
        if self.use_cache:
            cache_dict = self.neutral_cache if is_mp3 else self.emotion_cache
            if file_path in cache_dict:
                return cache_dict[file_path]
        # Check for mp3 or wav and load the waveform accordingly
        if is_mp3:
            waveform, sr = torchaudio.load(file_path, format="mp3")
        else:
            waveform, sr = torchaudio.load(file_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            #print('srrrrrrrrrrrrrrrrrr')
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            #print('shapeeeeeeeeeee')
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Trim or pad to max_length if specified
        if self.max_length is not None:
            #print('max_lengthhhhhhhhhhhhhhhhh')
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)
        
        # Cache the result if caching is enabled
        if self.use_cache:
            #print('cacheeeeee')
            cache_dict = self.neutral_cache if is_mp3 else self.emotion_cache
            cache_dict[file_path] = mel_spec_db
        
        
        return mel_spec_db
        
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        pair = self.file_pairs[idx]
        
        # Load neutral audio (mp3)
        neutral_spec = self._load_and_preprocess_audio(pair['neutral'], is_mp3=True)
        
        # Load emotional audio (wav)
        emotion_spec = self._load_and_preprocess_audio(pair['emotion'], is_mp3=False)
        
        # Convert emotion label to one-hot encoding
        emotion_label = torch.zeros(len(self.emotion_categories))
        emotion_label[pair['emotion_label']] = 1.0
        
        return {
            'neutral_spec': neutral_spec,
            'emotion_spec': emotion_spec,
            'emotion_label': emotion_label
        }
    
    def clear_cache(self):
        """Clear the cache to free memory"""
        self.neutral_cache.clear()
        self.emotion_cache.clear()
        print("Audio cache cleared")

def get_dataloaders(batch_size=BATCH_SIZE, train_ratio=0.8, use_cache=True):
    """Create train and validation dataloaders"""
    dataset = AudioEmotionDataset(use_cache=use_cache)
    
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    
    # Split into train and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    num_workers = 4
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True  # This can help with GPU transfer speed
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset