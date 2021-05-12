import torch
import speechbrain
from speechbrain.dataio.dataset import DynamicItemDataset

def create_datasets(hparams):
    datasets = {}

    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline(wav_files):
        predictor = speechbrain.dataio.dataio.read_audio_multichannel(wav_files['predictors'])
        target = speechbrain.dataio.dataio.read_audio(wav_files['wave_target'])

        return predictor.transpose(0, 1), torch.unsqueeze(target, 0)
    
    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor")
    def audio_pipeline_test(wav_files):
        predictor = speechbrain.dataio.dataio.read_audio_multichannel(wav_files['predictors'])
        return predictor.transpose(0, 1)
    
    for set_ in ['train', 'valid', 'test']:
        output_keys = ["id", "predictor", "target", "transcript"]
        dynamic_items=[audio_pipeline]

        if set_ == 'test':
            output_keys = ["id", "predictor"]
            dynamic_items=[audio_pipeline_test]

        datasets[set_] = DynamicItemDataset.from_json(
            json_path=hparams[f"{set_}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        ).filtered_sorted(sort_key="length")

    return datasets