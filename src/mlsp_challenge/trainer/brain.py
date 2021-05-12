import os
import torch
import numpy as np
import speechbrain as sb
from tqdm.contrib import tqdm
from torch.utils.data import DataLoader
from mlsp_challenge.losses import task1_metric

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        noisy_wavs, lens = batch.predictor
        out = self.modules.model(noisy_wavs)

        return out/out.max(2, keepdim=True)[0] * 0.9
    
    def compute_objectives(self, predictions, batch, stage):
        # Get clean targets
        targets, lens = batch.target

        # Compare the waveforms
        loss = sb.nnet.losses.mse_loss(predictions, targets)

        # Append this batch of losses to the loss metric
        self.loss_metric.append(
            batch.id, predictions, targets, reduction="batch"
        )

        # Since it is slow, only compute task1 metric on evalution sets
        if stage != sb.Stage.TRAIN:
            self.l3das_task1_metric.append(
                batch.id,
                np.squeeze(targets.cpu().numpy()),
                np.squeeze(predictions.cpu().numpy())
            )

        return loss
    
    def on_stage_start(self, stage, epoch=None):
        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.mse_loss
        )

        # Add a metric for evaluation sets
        if stage != sb.Stage.TRAIN:
            self.l3das_task1_metric = sb.utils.metric_stats.MetricStats(
                metric=task1_metric
            )

    def on_stage_end(self, stage, stage_loss, epoch=None):
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "task1_metric": self.l3das_task1_metric.summarize("average"),
            }

        # At the end of validation, we can write stats and checkpoints
        if stage == sb.Stage.VALID:
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            # unless they have the current best task1_metric
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["loss"])
    
    def predict(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar
        
        # Construct test data loader
        if not isinstance(test_set, DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        
        # Call the predict
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()

        with torch.no_grad():
            for batch in tqdm(
                test_set, dynamic_ncols=True, disable=not progressbar
            ):
                self.step += 1
                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                self.write_audios(batch.id, out)
        self.step = 0
    
    def write_audios(self, utt_id, batch):
        audio_path = self.hparams.audio_result
        npy_path = self.hparams.npy_result
        if not os.path.exists(audio_path):
            os.mkdir(audio_path)
        
        if not os.path.exists(npy_path):
            os.mkdir(npy_path)

        for count, utt in enumerate(utt_id):
            audio = batch[count, 0, :].detach().cpu()

            # Save as wav file to listen to the result
            sb.dataio.dataio.write_audio(
                filepath=os.path.join(
                    audio_path, 
                    "{}.wav".format(utt)
                ),
                audio=audio, 
                samplerate=16000
            )

            # Save as .npy for submission
            with open(
                os.path.join(npy_path, 
                "{}.wav".format(utt)), 'wb'
            ) as f:
                np.save(f, audio.numpy())