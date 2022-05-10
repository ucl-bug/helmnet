from helmnet import IterativeSolver, load_settings
from helmnet.support_functions import fig_generic
import numpy as np

def sample_prediction(use_cuda: bool = False) :
    settings = load_settings("experiments/base.json")

    solver = IterativeSolver.load_from_checkpoint(
        checkpoint_path="checkpoints/trained_weights.ckpt", strict=False, test_data_path=settings["medium"]["test_set"]
    )
    solver.freeze()  # To evaluate the model without changing it

    if use_cuda:
        solver.to("cuda:0")

    # Setup problem
    source_location = [30, 128]
    sos_map = np.ones((256, 256))
    sos_map[100:170, 30:240] = np.tile(np.linspace(2,1,210),(70,1))

    # Set model domain size (assumed square)
    solver.set_domain_size(sos_map.shape[-1], source_location=source_location)

    # Run example in kWave and pytorch, and produce figure
    fig_generic(
        solver,
        sos_map,
        path="images/withgmres",
        source_location=source_location,
        omega=1,
        min_sos=1,
        cfl=0.1,
        roundtrips=10.0,
        mode="normal",
    )

if __name__ == "__main__":
    sample_prediction()
