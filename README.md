# VoiceToDigit
Bot convertor

🎧 Telegram Bot: Digit Audio to Text Prediction

This project implements a Telegram bot that recognizes spoken digits (0–9) from voice messages.
It uses a trained Neural Network (Keras Sequential model), Librosa for audio preprocessing (mel-spectrograms), and pyTelegramBotAPI for Telegram integration.

🚀 Features

🎙️ Users send a voice message to the bot
🔊 Bot downloads the .ogg voice file and converts it to .wav
🎼 Converts audio into mel-spectrogram using Librosa
🧠 Feeds the spectrogram into a pre-trained Keras model
🔢 Predicts and replies with the spoken digit (0–9)
🛠️ Tech Stack

Python 3.10+
Keras for neural networks
Librosa for audio feature extraction
PyTelegramBotAPI for Telegram Bot
Pydub for audio format conversion
python-dotenv for .env configuration
📦 Installation

Clone the repository

git clone https://github.com/yourusername/telegram-digit-bot.git
cd telegram-digit-bot
Install dependencies

pip install -r requirements.txt
Create .env file

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
Download the trained model Place model.h5 in the root directory. You can train your own using FSDD (Free Spoken Digit Dataset).

Run the bot

python bot.py
📁 Project Structure

telegram-digit-bot/ │ ├── bot.py # Main bot logic (Telegram + audio processing) ├── model.h5 # Trained Keras model for digit recognition ├── preprocess.py # Audio to mel-spectrogram conversion ├── utils.py # Helper functions ├── requirements.txt ├── .env # Contains TELEGRAM_BOT_TOKEN (not versioned) └── README.md

🧠 Model Info

Dataset: Free Spoken Digit Dataset (FSDD)

Input: Mel-spectrogram (e.g. shape (128, 128))

Output: One of 10 classes (0–9)

Architecture Example:

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
🔊 Audio Processing

User sends .ogg voice message
Bot downloads it and converts it to .wav using pydub
librosa loads .wav, extracts mel-spectrogram
Spectrogram is reshaped and passed to the model
Prediction result is sent back to the user
📊 Example Interaction

User: Sends voice message saying "five" Bot:

🧠 Predicted digit: 5

✅ Environment Variables

Create a .env file in the root of the project:

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
Make sure not to commit this file!

🧪 Testing

To test the model manually:

from keras.models import load_model
from preprocess import audio_to_melspec

model = load_model('model.h5')
mel = audio_to_melspec('sample.wav')
prediction = model.predict(mel)
print(prediction.argmax())  # => 0–9
📌 TODO

 Add Docker support
 Host on Railway/Render/Heroku
 Support multilingual digits
 Add logging and error handling
📜 License

MIT License © Your Name

🙌 Acknowledgements

FSDD Dataset
Keras
Librosa
PyTelegramBotAPI
