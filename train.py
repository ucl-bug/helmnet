import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from helmnet import IterativeSolver, load_settings
import os
from argparse import ArgumentParser


if __name__ == "__main__":
    # Parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="ddp",
        help="Distributed training backend, see https://pytorch.org/tutorials/intermediate/ddp_tutorial.html.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="2,3,4,5,6,7",
        help="IDs of the GPUs to use during training, separated by a comma.",
    )
    parser.add_argument(    
        "--precision",
        type=int,
        default="32",
        help="Bits precision to use for calculations, can be either 32 or 16.",  #16 bit is volatile. when i tried it, NaNs everywhere all the time
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Number of total epochs for training.",
    )
    parser.add_argument("--track_arg_norm",             type=bool,  default=True)
    parser.add_argument("--terminate_on_nan",           type=bool,  default=True)
    parser.add_argument("--check_val_every_n_epoch",    type=int,   default=2)
    parser.add_argument("--limit_val_batches",          type=float, default=1.0)
    parser.add_argument("--num_sanity_val_steps",       type=int,   default=1)
    parser.add_argument("--benchmark",                  type=bool,  default=True)

    # Loading setings file
    settings = load_settings("experiments/base.json")

    # Making model
    solver = IterativeSolver(
        batch_size =                settings["training"]["train batch size"],
        domain_size =               settings["geometry"]["grid size"],
        k =                         settings["source"]["omega"] / settings["medium"]["c0"],
        omega =                     settings["source"]["omega"],
        gradient_clip_val =         settings["training"]["gradient clipping"],
        learning_rate =             settings["training"]["learning rate"],
        loss =                      settings["training"]["loss"],
        minimum_learning_rate =     settings["training"]["minimum learning rate"],
        optimizer =                 settings["training"]["optimizer"],
        PMLsize =                   settings["geometry"]["PML Size"],
        sigma_max =                 settings["geometry"]["sigma max"],
        source_location =           settings["source"]["location"],
        source_amplitude =          settings["source"]["amplitude"],
        source_phase =              settings["source"]["phase"],
        source_smoothing =          settings["source"]["smoothing"],
        train_data_path =           settings["medium"]["train_set"],
        validation_data_path =      settings["medium"]["validation_set"],
        activation_function =       settings["neural_network"]["activation function"],
        depth =                     settings["neural_network"]["depth"],
        features =                  settings["neural_network"]["channels per layer"],
        max_iterations =            settings["environment"]["max iterations"],
        state_channels =            settings["neural_network"]["state channels"],
        state_depth =               settings["neural_network"]["states depth"],
        weight_decay =              settings["training"]["weight_decay"],
    )

    # Create trainer
    logger = TensorBoardLogger("logs", name="helmnet")

    checkpoint_callback = ModelCheckpoint(
        dirpath     = os.getcwd() + "/checkpoints/",
        save_top_k  = 3,
        verbose     = True,
        monitor     = "val_loss",
        mode        = "min",
        save_last   = True,
    )

    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # Make trainer
    trainer = pl.Trainer.from_argparse_args(
        args, 
        logger      = logger, 
        callbacks   = [checkpoint_callback]
    )

    # Train network
    trainer.fit(solver)
