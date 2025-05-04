"""Conditional VAE model for audio emotion transfer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import LATENT_DIM, EMOTION_DIM

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants from original code
LATENT_DIM = 128
EMOTION_DIM = 2

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer for style/condition injection
    """
    def __init__(self, feature_dim, condition_dim):
        super(AdaIN, self).__init__()
        
        # Networks to predict scale and bias from emotion condition
        self.scale_transform = nn.Linear(condition_dim, feature_dim)
        self.bias_transform = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, x, condition):
        # x shape: [B, C, H, W]
        # condition shape: [B, condition_dim]
        
        # Instance normalization (without affine parameters)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.clamp(x.std(dim=(2, 3), keepdim=True), min=1e-5)
        normalized = (x - mean) / std
        
        # Generate adaptive scale and bias from condition
        scale = self.scale_transform(condition).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        bias = self.bias_transform(condition).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        
        # Apply the adaptive scaling and bias
        return scale * normalized + bias


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emotion_dim, kernel_size=3, 
                 stride=1, padding=1):
        super(EncoderBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.adain = AdaIN(out_channels, emotion_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, emotion):
        x = self.conv(x)
        x = self.adain(x, emotion)
        return self.relu(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emotion_dim, kernel_size=3, 
                 stride=1, padding=1, output_padding=0):
        super(DecoderBlock, self).__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        
        self.adain = AdaIN(out_channels, emotion_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x, emotion):
        x = self.conv_transpose(x)
        x = self.adain(x, emotion)
        return self.relu(x)


class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM, emotion_dim=EMOTION_DIM):
        """
        Conditional Variational Autoencoder for audio emotion transfer with AdaIN.
        
        Args:
            input_dim: Tuple of (height, width) for input spectrograms
            latent_dim: Dimension of latent space
            emotion_dim: Dimension of emotion encoding
        """
        super(CVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.emotion_dim = emotion_dim
        
        # Calculate flattened input size
        self.flat_size = input_dim[0] * input_dim[1]
        
        # Encoder blocks with AdaIN
        self.enc1 = EncoderBlock(1, 32, emotion_dim, stride=2)
        #self.enc2 = EncoderBlock(32, 32, emotion_dim)
        self.enc3 = EncoderBlock(32, 64, emotion_dim, stride=2)
        #self.enc4 = EncoderBlock(64, 64, emotion_dim)
        self.enc5 = EncoderBlock(64, 128, emotion_dim, stride=2)
        self.enc6 = EncoderBlock(128, 256, emotion_dim, stride=2)
        
        self.encoded_h = 5    # From debug output in original code
        self.encoded_w = 14   # From debug output in original code
        self.encoded_channels = 256
        
        # Calculate correct flattened encoding size
        self.encoding_size = self.encoded_channels * self.encoded_h * self.encoded_w
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.encoding_size + emotion_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoding_size + emotion_dim, latent_dim)
        
        # Decoder (from latent space to conv input size)
        self.decoder_input = nn.Linear(latent_dim + emotion_dim, self.encoding_size)
        
        # Decoder blocks with AdaIN for emotion conditioning
        self.dec1 = DecoderBlock(256, 128, emotion_dim, stride=2, output_padding=1)
        self.dec2 = DecoderBlock(128, 64, emotion_dim, stride=2, output_padding=1)
        #self.dec3 = DecoderBlock(64, 64, emotion_dim)
        self.dec4 = DecoderBlock(64, 32, emotion_dim, stride=2, output_padding=1)
        #self.dec5 = DecoderBlock(32, 32, emotion_dim)
        
        # Final output layer
        self.dec_output = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, 
                                            padding=1, output_padding=1)
    
    def encode(self, x, emotion):
        """
        Encode input to latent space with AdaIN conditioning
        
        Args:
            x: Input tensor [batch_size, 1, H, W]
            emotion: Emotion tensor [batch_size, emotion_dim]
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Apply encoder blocks with AdaIN conditioning
        h = self.enc1(x, emotion)
        #h = self.enc2(h, emotion)
        h = self.enc3(h, emotion)
        #h = self.enc4(h, emotion)
        h = self.enc5(h, emotion)
        h = self.enc6(h, emotion)
        
        h_flat = self.flatten(h)
        
        # Concatenate with emotion encoding
        h_emotion = torch.cat([h_flat, emotion], dim=1)
        
        # Get latent parameters
        mu = self.fc_mu(h_emotion)
        logvar = self.fc_logvar(h_emotion)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, emotion):
        """
        Decode latent vector to output space with AdaIN conditioning
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            emotion: Emotion tensor [batch_size, emotion_dim]
            
        Returns:
            output: Reconstructed output
        """
        # Concatenate latent vector with emotion
        z_emotion = torch.cat([z, emotion], dim=1)
        
        # Decode to flattened encoder output size
        h = self.decoder_input(z_emotion)
        
        # Reshape to match encoder output dimensions before transposed convs
        h = h.view(-1, self.encoded_channels, self.encoded_h, self.encoded_w)
        
        # Apply decoder blocks with AdaIN conditioning
        h = self.dec1(h, emotion)
        h = self.dec2(h, emotion)
        #h = self.dec3(h, emotion)
        h = self.dec4(h, emotion)
        #h = self.dec5(h, emotion)
        
        # Final output layer
        output = self.dec_output(h)
        
        # Ensure output has the same dimensions as input using interpolation if needed
        if output.shape[2:] != self.input_dim:
            output = F.interpolate(output, size=self.input_dim, mode='bilinear', align_corners=False)
            
        return output
    
    def forward(self, x, emotion):
        """
        Forward pass
        
        Args:
            x: Input tensor
            emotion: Emotion tensor
            
        Returns:
            output: Reconstructed output
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x, emotion)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, emotion)
        return output, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Combined reconstruction + KL divergence loss
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean from latent space
        logvar: Log variance from latent space
        beta: Weight for KL divergence term
        
    Returns:
        loss: Total loss
    """
    # MSE reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    return recon_loss + beta * kl_loss