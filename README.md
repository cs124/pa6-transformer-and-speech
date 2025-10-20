#  Programming Assignment 6 - Transformers and Speech
  
In this assignment, you will implement a Transformer-based language model for text generation and train it on a dataset of Shakespeare's plays. 
You will then use your trained model to generate some texts, convert them into audio files, and transcribe the audio files back into text.

## Environment Setup

```
conda create -n cs124 python=3.10 -y
conda activate cs124
pip install "torch==2.6.0" "numpy==2.1.3" "transformers==4.39.3" "datasets==2.20.0" "tiktoken-0.7.0" "wandb==0.17.6" "tqdm==4.66.4"
```

## Outline 

Data have already been preprocessed into a format that can be used to train a GPT-2 model. Your tasks are:

TODO: add torch mini tutorial

- Implementing attention in Transformer (model.py)
- Running the provided training script (train.py)
- Sample a few sentences from the trained model (sample.py)
- Check perplexity on some test sentences we provide? 
- Convert your sampled sentences into audio with OpenAI TTS API (tts.py)
- TODO: ask them to record them saying those sentences in noisy environments
- Convert your generated audio back to text with Whisper (asr.py)
- Just ask them to upload wav file and texts.

## Part 1: Implement Attention in Transformer and Train the Model

To train the model on the Shakespeare dataset, run:

```
python train.py
```

To sample from the trained model, run:

```
python sample.py
```

You can tweak the hyperparameters in both the training and sampling scripts. As part of the assignment, you need to submit 5 sampled sentences from the trained model with reasonable quality (i.e., not gibberish).