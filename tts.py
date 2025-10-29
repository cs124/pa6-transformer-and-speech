from pathlib import Path
from together import Together

client = Together()

speech_file_path = "speech.mp3"

response = client.audio.speech.create(
    model="cartesia/sonic",
    input="Today is a wonderful day to build something people love!",
    voice="helpful woman",
)

response.stream_to_file(speech_file_path)
