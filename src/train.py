import sys
import datetime
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from mlsp_challenge.trainer import SEBrain
from mlsp_challenge.data import create_datasets
from mlsp_challenge.data import prep_librispeech

def main(hparams, hparams_file, run_opts, overrides):
    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data prep to run on the main thread
    sb.utils.distributed.run_on_main(
        prep_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "train_folder": hparams["train_folder"],
            "valid_folder": hparams["valid_folder"],
        },
    )

    # Create dataset objects "train" and "valid"
    datasets = create_datasets(hparams)

    # # Initialize the Trainer
    se_brain = SEBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Call the training loop
    se_brain.fit(
        epoch_counter=se_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    return None

if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    main(hparams, hparams_file, run_opts, overrides)