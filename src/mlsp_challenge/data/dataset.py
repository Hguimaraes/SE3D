import torch
import speechbrain
from speechbrain.dataio.dataset import DynamicItemDataset

def create_datasets(hparams):
    datasets = {}

    """
    Pad an audio to a power of 2**depth
    necessary for decimate operation in the network
    """
    def pad_power(audio):
        depth = hparams['m_depth']
        power = 2**depth-1
        n = audio.shape[0]
        audio = audio.transpose(0, 1)

        if n % (power + 1) != 0:
            diff = (n|power) + 1 - n
            audio = torch.nn.functional.pad(audio, (0, diff))

        return audio


    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline(wav_files):
        max_size = hparams['max_train_sample_size']
        predictor = speechbrain.dataio.dataio.read_audio_multichannel(
            wav_files['predictors']
        )

        target = speechbrain.dataio.dataio.read_audio(wav_files['wave_target'])
        target = torch.unsqueeze(target, -1)

        # sampling process
        samples = predictor.shape[0]
        if samples > max_size:
            offset = torch.randint(low=0, high=max_size-1, size=(1,))
            target = target[offset:(offset+max_size), :]
            predictor = predictor[offset:(offset+max_size), :]
        
        return pad_power(predictor), pad_power(target)


    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline_valid(wav_files):
        max_size = hparams['max_train_sample_size']
        predictor = speechbrain.dataio.dataio.read_audio_multichannel(
            wav_files['predictors']
        )

        target = speechbrain.dataio.dataio.read_audio(wav_files['wave_target'])
        target = torch.unsqueeze(target, -1)

        return pad_power(predictor), pad_power(target)


    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor")
    def audio_pipeline_test(wav_files):
        predictor = speechbrain.dataio.dataio.read_audio_multichannel(
            wav_files['predictors']
        )

        return predictor.transpose(0, 1)
    
    dynamic_items_map = {
        'train': [audio_pipeline],
        'valid': [audio_pipeline_valid],
        'test': [audio_pipeline_test]
    }

    for set_ in ['train', 'valid', 'test']:
        output_keys = ["id", "predictor"]
        dynamic_items = dynamic_items_map[set_]

        if set_ is not 'test':
            output_keys.extend(["target", "transcript"])
        
        # Construct the dynamic item dataset
        datasets[set_] = DynamicItemDataset.from_json(
            json_path=hparams[f"{set_}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        ).filtered_sorted(sort_key="length")

    return datasets