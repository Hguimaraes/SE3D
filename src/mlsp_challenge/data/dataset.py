import torch
import speechbrain
from speechbrain.dataio.dataset import DynamicItemDataset

def create_datasets(hparams):
    datasets = {}

    @speechbrain.utils.data_pipeline.takes("wav_files")
    @speechbrain.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline(wav_files):
        sig_a = speechbrain.dataio.dataio.read_audio(wav_files['wave_a'])
        sig_b = speechbrain.dataio.dataio.read_audio(wav_files['wave_b'])
        predictor = torch.cat([sig_a, sig_b], dim=1)
        target = speechbrain.dataio.dataio.read_audio(wav_files['wave_target'])
        
        return predictor, target
    
    for set_ in ['train', 'valid']:
        datasets[set_] = DynamicItemDataset.from_json(
            json_path=hparams[f"{set_}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline],
            output_keys=["predictor", "target", "transcript"],
        ).filtered_sorted(sort_key="length")

    return datasets