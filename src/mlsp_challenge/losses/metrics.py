import os
import torch
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
wer_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h");
wer_model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h");

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
    try:   #if no words are predicted
        wer_val = jiwer.wer(transcript[0], transcript[1])
    except ValueError:
        wer_val = None

    return wer_val

def task1_metric(clean_speech_array, denoised_speech_array, sr=16000):
    '''
    Compute evaluation metric for task 1 as (stoi+(1-word error rate)/2)
    This function computes such measure for 1 single datapoint
    '''
    WER = 0.
    STOI = 0.
    METRIC = 0.
    len_samples = clean_speech_array.shape[0]

    for example_num, (clean_speech, denoised_speech) in enumerate(zip(clean_speech_array, denoised_speech_array)):
        wer_metric = wer(clean_speech, denoised_speech)
        if wer_metric is not None:  #if there is no speech in the segment
            # metric of the sample
            stoi_metric = stoi(clean_speech, denoised_speech, sr, extended=False)
            wer_metric = np.clip(wer_metric, 0., 1.)
            stoi_metric = np.clip(stoi_metric, 0., 1.)
            metric = (stoi_metric + (1. - wer_metric)) / 2.

            # metric of the batch
            METRIC += (1. / float(example_num + 1)) * (metric - METRIC)
            WER += (1. / float(example_num + 1)) * (wer_metric - WER)
            STOI += (1. / float(example_num + 1)) * (stoi_metric - STOI)

        else:
            metric = None
            STOI = None

    return torch.Tensor([METRIC])

def compute_se_metrics(predicted_folder, truth_folder, fs=16000):
    '''
    Load all submitted sounds for task 1 and compute the average metric
    '''
    METRIC = []
    WER = []
    STOI = []
    predicted_list = [s for s in os.listdir(predicted_folder) if '.wav' in s]
    truth_list = [s for s in os.listdir(truth_folder) if '.wav' in s]
    n_sounds = len(predicted_list)
    for i in range(n_sounds):
        name = str(i) + '.wav'
        predicted_temp_path = os.path.join(predicted_folder, name)
        truth_temp_path = os.path.join(truth_folder, name)
        predicted = librosa.load(predicted_temp_path, sr=fs)
        truth = librosa.load(truth_temp_path, sr=fs)
        metric, wer, stoi = task1_metric(truth, predicted)
        METRIC.append(metric)
        WER.append(wer)
        STOI.append(stoi)

    average_metric = np.mean(METRIC)
    average_wer = np.mean(WER)
    average_stoi = np.mean(STOI)

    print ('*******************************')
    print ('Task 1 metric: ', average_metric)
    print ('Word error rate: ', average_wer)
    print ('Stoi: ', average_stoi)

    return average_metric