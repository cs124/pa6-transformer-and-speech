from pathlib import Path
from together import Together
import json 

client = Together()

sampled_sentences = json.load(open('sampled_sentences.json'))
all_sentences = "\n".join(sampled_sentences)

response = client.audio.speech.create(
    model="cartesia/sonic",
    input=all_sentences,
    voice="helpful woman",
)
response.stream_to_file("sampled_sentences_speech.mp3")
