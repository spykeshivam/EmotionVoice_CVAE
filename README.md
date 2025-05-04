# ⚠️ **Work in Progress - Expect Rough Edges!** ⚠️

## Project description

This project develops a pipeline for converting text into emotionally expressive speech, with
a focus on “Happy” and “Angry” emotions. To achieve this, we use 2 models: First model
used here is google’s Text to speech model.[6] It takes an input text which is converted into
neutral speech audio using Google’s gTTS API. Next, we perform emotion conversion via
audio-domain style transfer in the audio domain. In this stage, given the audio file of neutral
speech, we learn to apply an “emotional style” (happiness or anger). To do this, we
implement a Conditional Variational Autoencoder (cVAE) in PyTorch that operates directly on
mel-spectrogram representations. By conditioning on discrete emotion labels, our cVAE tries
to learn to disentangle content and style in its latent space and to reconstruct neutral
spectrograms into their emotionally styled counterparts.

## Datasets used

Since we use pretrained model for TTS, we are not training it on any dataset. To train the
2nd model(cVAE), we use the Audio emotions[7] dataset. The dataset contains audio files of
variable length collected and combined from multiple sources for 7 different emotions. We
filter out the audio files for “Happy” and “Angry” emotions. Our cVAE requires 3 inputs to
train and learn the emotions: Neutral audio spectrogram, Emotional audio spectrogram and
Emotion label(Happy or sad). From the audio emotions dataset we already have Emotional
audio spectrogram and Emotion label(Happy or sad). This provided us with 2167 records of
each label. (Total 4334 records). But we still need the neutral audio spectrogram
corresponding to each emotional audio spectrogram.

## Current Status

This project is in its early stages of development. The current outputs are considered preliminary and significantly impacted by data alignment challenges.

### Known Limitations

* **Data and Alignment Issues:** A major challenge currently hindering the performance of the conditional Variational Autoencoder (cVAE) is the lack of guaranteed lexical or temporal alignment between the neutral (synthesized via gTTS) and emotional audio clip pairs. This misalignment is a primary reason for the poor output quality at this stage.

### Proposed Solution

We are planning to address this issue by implementing Dynamic Time Warping (DTW). This technique will be explored soon to improve the alignment between the audio clips.




