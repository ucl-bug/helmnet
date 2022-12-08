from helmnet import IterativeSolver
import numpy as np
import torch
from matplotlib import pyplot as plt

# Setup the solver, loading the pre-trained weights.
solver = IterativeSolver.load_from_checkpoint("checkpoints/trained_weights.ckpt", strict=False, test_data_path=None)

# Freeze the model to speed it up, as we are not changing it.
solver.freeze()

# Define if the model should run on the CPU ("cpu") or GPU (e.g., "cuda:0").
solver.to("cuda:0")

# Setup sound speed map defined as a numpy array.
sos_map = np.ones((256, 256))
sos_map[100:170, 30:240] = 1.5

# Define the source. This can be either a [x, y] position, e.g., source_location = [30, 128]
# or as a spatial map. Here we use a spatial map. Both real and imaginary components can
# be defined as separate channels. Here we only define the real part.
source_map = np.zeros((2, 256, 256))
source_map[0, 30, 120:130] = 1

# The sound speed and source inputs must be defined as four-dimensional pytorch tensors.
# The first dimension is the batch dimension. This can be used to run multiple examples at the same time. 
# The second dimension is the channel dimension. For sound speed, this isn't used. For sources,
# this is the real and imaginary components.
sos_map_tensor = (torch.tensor(sos_map).unsqueeze(0).unsqueeze(0).to(solver.device)).float()
source_map_tensor = (torch.tensor(source_map).unsqueeze(0).to(solver.device)).float()
print("Sos tensor shape: ", sos_map_tensor.size())

# Set model domain size and source.
# To use [x, y] position, instead use source_location=source_location.
solver.set_domain_size(sos_map.shape[-1], source_map=source_map_tensor)

# Disable gradient calculation for inference (reduces memory consumption).
with torch.no_grad():

    # Run model with 100 iterations.
    output = solver.forward(sos_map_tensor, num_iterations=100)
    print(type(output))

    # The output wavefield is indexed as (iteration index, batch index, real/imaginary, Nx, Ny).
    # In this case, the size is (1, 1, 2, 256, 256), and we take just the real part.
    p_real = np.asarray(output["wavefields"][0][0][0].cpu())

    # Plot the wavefield
    plt.figure(figsize=(8, 6))
    plt.imshow(p_real, vmin=-0.5, vmax=0.5, cmap="seismic")
    plt.colorbar()
    plt.show()
