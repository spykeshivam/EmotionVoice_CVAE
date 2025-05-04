BELOW ARE THE DESCRIPTION OF ALL THE FILES AND FOLDERS. I WAS NOT ABLE TO UPLOAD THE AUDIO DATASETS BECAUSE OF THEIR FILE SIZE.

main.ipynb : Training code and model inference.

config.py : Config file

datasets: Contains the 4 folders : Angry, Happy are from the Audio emotion dataset. Audio_padded and Happy_padded are the audio files from the Audio emotion dataset padded to 5 seconds.

Text : Angry, Happy are from the Audio emotion dataset transcribes to text using openAI whisper

Audio : Contains audio files converted using gtts from the text files in the Text folder

Audio_transformed : Contains audio files after pitch shifting on the Audio folder

outputs : Stores the model and files while processing main.ipynb

src : Contains codes with helper functions for ipynb

inference.py : Contains codes with helper functions for ipynb

Speech_to_text.ipynb : Dataset creation and gtts evaluation code

Text_evaluation_tts : Text for 200 files for gtts evaluation


deleted_files_log.csv : Deleted files for being > 5 seconds

deleted_non_english_files.csv : Deleted non english files 




