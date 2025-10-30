from together import Together

## Initialize the client

client = Together()

## Basic transcription

response = client.audio.transcriptions.create(
    file="sampled_sentences_speech.mp3",
    model="openai/whisper-large-v3",
    language="en",
)
print(response.text)

# save the transcription to a txt file
with open('sampled_sentences_speech.txt', 'w') as f:
    f.write(response.text)
