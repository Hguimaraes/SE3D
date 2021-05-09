import torch
import numpy as np
import speechbrain as sb
from mlsp_challenge.losses import task1_metric

class SEBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        return self.modules.model(batch.predictor.data)
    
    def compute_objectives(self, predictions, batch, stage):
        # Get clean targets
        targets = batch.target.data

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
            # unless they have the current best STOI score.
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["loss"])
