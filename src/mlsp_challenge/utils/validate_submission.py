import os, sys
import numpy as np
import pandas as pd
import librosa
'''
Check if the the submssion folders are valid: all files must have the
correct format, shape and naming.
WORK IN PROGRESS...
'''

def validate_task1_submission(submission_folder, test_folder):
    '''
    Args:
    - submission_folder: folder containing the model's output for task 1 (non zipped).
    - test_folder: folder containing the released test data (non zipped).
    '''
    #read folders
    contents_submitted = sorted(os.listdir(submission_folder))
    contents_test = sorted(os.listdir(test_folder))
    contents_submitted = [i for i in contents_submitted if 'DS_Store' not in i]
    contents_test = [i for i in contents_test if 'DS_Store' not in i]
    contents_test = [i for i in contents_test if '_B' not in i]
    contents_test = [i.split('_')[0]+'.wav' for i in contents_test]

    #check if non.npy files are present
    non_npy = [x for x in contents_submitted if x[-4:] != '.npy']  #non .npy files
    if len(non_npy) > 0:
        raise AssertionError ('Non-.npy files present. Please include only .npy files '
                              'in the submission folder.')

    #check total number of files
    num_files = len(contents_submitted)
    target_num_files = len(contents_test)
    if not num_files == target_num_files:
        raise AssertionError ('Wrong amount of files. Target:' + str(target_num_files) +
                             ', detected:' + str(len(contents_submitted)))

    #check files naming
    names_submitted = [i.split('.')[0] for i in contents_submitted]
    names_test = [i.split('.')[0] for i in contents_test]
    names_submitted.sort()
    names_test.sort()
    if not names_submitted == names_test:
        raise AssertionError ('Wrong file naming. Please name each output file '
                               'exactly as its input .wav file, but with .npy extension')

    #check shape file-by-file
    for i in contents_test:
        submitted_path = os.path.join(submission_folder, i.split('.')[0]+'.npy')
        test_path = os.path.join(test_folder, i.split('.')[0]+'_A.wav')
        s = np.load(submitted_path, allow_pickle=True)
        t, _ = librosa.load(test_path, 16000, mono=False)
        target_shape = t.shape[-1]
        if not s.shape[-1] == target_shape:
            raise AssertionError ('Wrong shape for: ' + str(i) + '. Target: ' + str(target_shape) +
                                 ', detected:' + str(s.shape))

    print ('The shape of your submission for Task 1 is valid!')