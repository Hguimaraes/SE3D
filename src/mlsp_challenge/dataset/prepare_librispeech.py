import re
import glob
import json
import logging
import torchaudio
from tqdm import tqdm
from typing import List
from pathlib import Path
from speechbrain.utils.data_utils import get_all_files

logger = logging.getLogger(__name__)

"""
Set folders for train and test and prepare to create Json files.
"""
def  prep_librispeech(
    data_folder:str,
    save_json_train:str,
    save_json_valid:str,
    train_folder:str,
    valid_folder:str
    
):
    # File extension to look for
    extension = ["_A.wav"]

    # Parameters to search and save
    files = [
        [train_folder, extension, save_json_train],
        [valid_folder, extension, save_json_valid],
    ]

    for folder, exts, save_json in files:
        a_files = get_all_files(folder, match_and=exts)

        # Create Json for dataio
        create_json(a_files, save_json)

"""
Extracts info from the 4 files associated with each sample.
The predictors data of this section are released as 8-channels 
16kHz 16 bit wav files, consisting of 2 sets of first-order 
Ambisonics recordings.
file_A.wav: Predictor - Channels [WA,ZA,YA,XA] 
file_B.wav: Predictor - Channels [WB,ZB,YB,XB]
    where A/B refers to the used microphone 
    and WXYZ are the b-format ambisonics channels.

file.wav: Mono wave from Librispeech without noise. SE target.
file.txt: Transcripts for file.wav. We are going to perform WER on this.
"""
def create_json(
    a_wav_list:List[str],
    json_file:str
):
    # Processing all the wav files in the list
    json_dict = {}
    for utterance in tqdm(a_wav_list):
        # Manipulate paths and get info about the 4 files
        utt_id = Path(utterance).stem[:-2]
        label_utt = re.sub('/data/', '/labels/', utterance)[:-6]

        utterance_b = utterance.replace("_A", "_B")
        transcript_path = label_utt + '.txt'
        wav_target = label_utt + '.wav'

        with open(transcript_path) as f:
            transcript = f.readline().replace('\n', '')

        # Construct Json structure
        json_dict[utt_id] = {
            "waves": {
                "wave_a": utterance,
                "wave_b": utterance_b,
                "wave_target": wav_target,
            },
            "lengths": {
                "len_wave_a": torchaudio.info(utterance).num_frames,
                "len_wave_b": torchaudio.info(utterance_b).num_frames,
                "len_wave_target": torchaudio.info(wav_target).num_frames
            },
            "speaker_ID": utt_id.split("-")[0],
            "transcript": transcript
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")