import sys
import datetime
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from mlsp_challenge.trainer import SEBrain
from mlsp_challenge.dataset import prep_librispeech

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

    # # Create dataset objects "train" and "valid"
    # datasets = dataio_prep(hparams)

    # # Initialize the Brain object to prepare for mask training.
    # se_brain = SEBrain(
    #     modules=hparams["modules"],
    #     opt_class=hparams["opt_class"],
    #     hparams=hparams,
    #     run_opts=run_opts,
    #     checkpointer=hparams["checkpointer"],
    # )

    # # The `fit()` method iterates the training loop, calling the methods
    # # necessary to update the parameters of the model. Since all objects
    # # with changing state are managed by the Checkpointer, training can be
    # # stopped at any point, and will be resumed on next call.
    # se_brain.fit(
    #     epoch_counter=se_brain.hparams.epoch_counter,
    #     train_set=datasets["train"],
    #     valid_set=datasets["valid"],
    #     train_loader_kwargs=hparams["dataloader_options"],
    #     valid_loader_kwargs=hparams["dataloader_options"],
    # )

    # # Load best checkpoint (highest STOI) for evaluation
    # test_stats = se_brain.evaluate(
    #     test_set=datasets["test"],
    #     max_key="stoi",
    #     test_loader_kwargs=hparams["dataloader_options"],
    # )
    return None

if __name__ == "__main__":
    # Reading command line arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as f:
        hparams = load_hyperpyyaml(f, overrides)
    
    main(hparams, hparams_file, run_opts, overrides)