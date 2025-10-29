from together import Together

## Initialize the client

client = Together()

## Basic transcription

response = client.audio.transcriptions.create(
    file="speech.mp3",
    model="openai/whisper-large-v3",
    language="en",
)
print(response.text)
