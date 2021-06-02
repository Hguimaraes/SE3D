import os
import torch
import time
import jiwer
import librosa
import warnings
import numpy as np
from pystoi import stoi
import transformers
from transformers import Wav2Vec2ForMaskedLM
from transformers import Wav2Vec2Tokenizer

'''
Functions to compute the metrics for the 2 tasks of the L3DAS21 challenge.
- task1_metric returns the metric for task 1.
- location_sensitive_detection returns the metric for task 1.
Both functions require numpy matrices as input and can compute only 1 batch at time.
Please, have a look at the "evaluation_baseline_taskX.py" scripts for detailed examples
on the use of these functions.
'''
warnings.filterwarnings("ignore", category=FutureWarning)
transformers.logging.set_verbosity_error()
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")

def wer(clean_speech, denoised_speech):
    """
    computes the word error rate(WER) score for 1 single data point
    """
    def _transcription(clean_speech, denoised_speech):

        # transcribe clean audio
        input_values = wer_tokenizer(clean_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_clean = wer_tokenizer.batch_decode(predicted_ids)[0];

        # transcribe
        input_values = wer_tokenizer(denoised_speech, return_tensors="pt").input_values;
        logits = wer_model(input_values).logits;
        predicted_ids = torch.argmax(logits, dim=-1);
        transcript_estimate = wer_tokenizer.batch_decode(predicted_ids)[0];

        return [transcript_clean, transcript_estimate]

    transcript = _transcription(clean_speech, denoised_speech);
    try:
        wer_val = jiwer.wer(*transcript)
    except ValueError:
        wer_val = None

    return wer_val

def task1_metric(clean_speech, denoised_speech, sr=16000):
    '''
    Compute evaluation metric for task 1 as (stoi+(1-word error rate)/2)
    This function computes such measure for 1 single datapoint
    '''
    WER = np.mean([wer(c, d) for c, d in zip(clean_speech, denoised_speech)])

    if WER is not None:
        STOI = np.mean([stoi(c, d, sr, extended=False) for c, d in zip(clean_speech, denoised_speech)])
        METRIC = (STOI + (1. - WER)) / 2.
    else:
        METRIC = None

    return torch.Tensor([METRIC])