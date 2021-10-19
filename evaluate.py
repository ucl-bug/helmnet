from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
from scipy.io import savemat
from torch.utils.data import DataLoader

from helmnet import IterativeSolver
from helmnet.dataloaders import get_dataset


class Evaluation:
    def __init__(self, path, testset, gpus):
        self.path = path
        self.testset = get_dataset(testset)
        self.testloader = DataLoader(
            self.testset, batch_size=32, num_workers=32, shuffle=False
        )
        self.gpus = gpus
        self.model = self.get_model()
        self.model.eval()
        self.model.freeze()

    def move_model_to_gpu(self):
        self.model.to("cuda:" + str(self.gpus[0]))

    def results_on_test_set(self):
        trainer = pl.Trainer(gpus=self.gpus)
        trainer.test(self.model, self.testloader)

    def compare_to_gmres(self):
        # self.testset.dataset.save_for_matlab('testset.mat')
        savemat("test_indices.mat", {"test_indices": np.array(self.testset.indices)})

    def single_example(self, idx, get_wavefield=True, get_states=True, iterations=1000):
        sos_map = self.testset[idx].unsqueeze(0).to("cuda:" + str(self.gpus[0]))

        output = self.model.forward(
            sos_map,
            num_iterations=iterations,
            return_wavefields=get_wavefield,
            return_states=get_wavefield,
        )
        # Get loss
        losses = [self.model.test_loss_function(x) for x in output["residuals"]]
        return output, losses

    def get_model(self, domain_size=None, source_location=None):

        # Loading model and its hyperparams
        model = IterativeSolver.load_from_checkpoint(self.path, strict=False, test_data_path=None)
        hparams = model.hparams

        # Customizing hparams if needed
        if domain_size is not None:
            hparams["domain_size"] = domain_size
        if source_location is not None:
            hparams["source_location"] = source_location
        new_model = IterativeSolver(**hparams)

        # loading weights and final setup
        new_model.f.load_state_dict(model.f.state_dict())

        new_model.set_laplacian()
        new_model.set_source()
        new_model.freeze()

        print("--- MODEL HYPERPARAMETERS ---")
        print(new_model.hparams)

        return new_model

    def set_domain_size(self, domain_size, source_location=None, source_map=None):
        self.model.hparams.domain_size = domain_size
        self.model.f.domain_size = self.model.hparams.domain_size
        self.model.set_laplacian()
        if source_location is not None:
            self.model.set_multiple_sources([source_location])
        else:
            self.model.set_source_maps(source_map)
        self.model.f.init_by_size()
        for enc, size in zip(self.model.f.enc, self.model.f.states_dimension):
            enc.domain_size = size


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="checkpoints/trained_weights.ckpt",
        help="Checkpoint file with model weights",
    )

    parser.add_argument(
        "--test_set",
        type=str,
        default="datasets/splitted_96/testset.ph",
        help="Test-set file",
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="Which gpu to use",
    )

    args = parser.parse_args()

    evaluator = Evaluation(
        path=args.model_checkpoint, testset=args.test_set, gpus=[args.gpu]
    )

    # Making results on the test set
    evaluator.results_on_test_set()
