#  Programming Assignment 6 - Transformers and Speech
  
In this assignment, you will implement part of a Transformer-based language model for text generation and train it on a dataset of Shakespeare corpus. 
You will then use your trained model to generate some texts, convert them into audio files, and transcribe the audio files back into text.

## Environment Setup

```
conda create -n cs124 python=3.10 -y
conda activate cs124
pip install "torch==2.6.0" "numpy==2.1.3" "transformers==4.39.3" "datasets==2.20.0" "tiktoken-0.7.0" "wandb==0.17.6" "tqdm==4.66.4" "openai==2.6.0" "together"
```

## Part 1: Implement Attention in Transformer

Check out `model.py` which implements a vanilla Transformer model. 

You need to implement the `forward` method of the `CausalSelfAttention` class based on what you learned in class. You do not need to change anything else in `model.py`, but you are encouraged to read through the implementation to understand the model architecture.

We have provided a simple unit test for the `CausalSelfAttention` class. You can test whether your implementation is correct by running:
```
python test_attention.py
```

## Part 2: Training some nanoGPT models 

Once that's done, you can train the model by running:

```
python train.py
```

We have already processed the Shakespeare corpus and stored the tokenized data in `data/`.
By default, the training script uses the data that's tokenized via the GPT-2 BPE tokenizer.
If you wish, you could also try training a character-level model by changing the `dataset` variable in `train.py` to `'shakespeare-char'`; although note that all of our benchmarking numbers are based on the BPE-tokenized data.
The training script runs on CPU by default. Running the training script for 2000 iterations on my Macbook Pro with M1 chip takes around 15 minutes and gets a loss below 4.0.
If you have extra time, you are encouraged to try out different hyperparameters and try to improve the validation loss of your model.

The trained model checkpoints will be automatically saved in the `out-shakespeare/` or `out-shakespeare-char/` directories depending on whether you trained a character-level model.
You can then load the model checkpoint and sample from the model by running:

```
python sample.py
```

You are encouraged to tweak the sampling hyperparameters to understand the effects of different sampling hyperparameters like the temperature. Our model is trained on a tiny corpus, so the sampled sentences may not be very fluent, but you should be able to see some Shakespearean-like writing patterns and some basic grammar. If you only get gibberish, you probably did something wrong either in the training or sampling process.

Use your trained model to sample 5 sentences with maximum 100 tokens each. As part of the submission, save your sampled sentences as a list of strings in a json file called `sampled_sentences.json`. We will judge the quality of your sampled sentences based on whether they are at least somewhat coherent and grammatically correct.

## Part 3: Compute Perplexity on Test Sentences

Once you feel confident about your models and sampled sentences, you should run the `perplexity.py` script to compute the perplexity of our test sentences with your trained models. The script will automatically save the results into a json file called `perplexity_results.json`. Do not modify the test sentences or the saving format because our grading script depends on it.

## Part 4: Convert Sampled Sentences into Audio Files

Next, you should use the Together API to convert your sampled sentences into audio files. You can do so by running the `tts.py` script. This script will automatically save the audio file as a mp3 file called `sampled_sentences_speech.mp3`. Remember that you need to set your Together API key in the `TOGETHER_API_KEY` environment variable, by example, by running:
```
export TOGETHER_API_KEY=your_api_key
```
before running the `tts.py` script.

Read the [Together API documentation](https://docs.together.ai/docs/text-to-speech) to understand how to use the API and try out different voices that they support.

## Part 5: Transcribe the Audio Files into Text

Next, you should use the Together API to transcribe the audio files into text. You can do so by running the `speech_to_text.py` script. This script will automatically save the text file as a txt file called `sampled_sentences_speech.txt`. Remember that you need to set your Together API key in the `TOGETHER_API_KEY` environment variable, by example, by running:
```
export TOGETHER_API_KEY=your_api_key
```
before running the `speech_to_text.py` script.

We will include the transcribed file `sampled_sentences_speech.txt` in the submission as part of the grading. You are also encouraged to try what happens when you record yourself reading the sampled sentences in a noisy environment and transcribe the audio file back into text; and compare the quality of the transcription with the original sampled sentences.

## Part 6: Zip and Submit 

Run `bash create_assignment_zip.sh` to zip your submission and submit the zip file to Gradescope.

To recap, the submission zip should include the following files:

- `model.py`: your implementation of the Transformer model
- `perplexity_results.json`: the perplexity results of your sampled sentences
- `sampled_sentences.json`: your sampled sentences
- `sampled_sentences_speech.txt`: the transcribed text of the audio file of your sampled sentences
