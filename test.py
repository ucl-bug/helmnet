from helmnet import IterativeSolver
from helmnet.support_functions import fig_generic
import numpy as np
import torch

solver = IterativeSolver.load_from_checkpoint(
    "checkpoints/trained_weights.ckpt", strict=False
)
solver.freeze()  # To evaluate the model without changing it
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
