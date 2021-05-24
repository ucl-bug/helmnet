import numpy as np
import torch
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import subprocess
from scipy.io import loadmat, savemat


def last_frame_difference(stream, reference, mask=None):
    with torch.no_grad():
        pytorch_wf = stream[:, -1, 0] + 1j * stream[:, -1, 1]
        stream = torch.tensor(pytorch_wf)
        reference = torch.tensor(reference)
        difference, normstream, norm_reference = difference_to_kwave(
            stream, reference, mask=mask
        )
        l_infty, indices = (difference).reshape(difference.shape[0], -1).topk(1, 1)
        mse = difference.pow(2).mean([1, 2]).sqrt()
    return l_infty[:, 0], mse


def difference_to_kwave(sample, reference, mask=None, pml_size=10):
    # Normalizing to source wavefield
    sample = sample / sample[:, 82, 48].unsqueeze(1).unsqueeze(1)
    if torch.any(torch.isnan(sample)):
        sample[torch.isnan(sample)] = 0.0
    reference = reference / reference[:, 82, 48].unsqueeze(1).unsqueeze(1)
    reference = torch.conj(reference)

    # Normalize error by maximum
    if mask is not None:
        sample = sample * mask
        reference = reference * mask
        max_vals = (
            torch.tensor([x.max() for x in reference.abs()])
            .unsqueeze(1)
            .unsqueeze(1)
            .to(reference.device)
        )
    else:
        max_vals = 1
    return (
        torch.abs(sample - reference)[:, pml_size:-pml_size, pml_size:-pml_size]
        / max_vals,
        sample,
        reference,
    )


def get_model_errors(pytorch_tensors, kwave_results, iterations=1000):
    print("Getting model error curves...")

    if os.path.isfile("results/model_traces.npz"):
        npzfile = np.load("results/model_traces.npz")
        return npzfile["l_infty_traces"], npzfile["mse_traces"]

    print("File not found: generating curves")

    mse_vs_iteration = []
    l_infty_vs_iteration = []

    for k in tqdm(range(50)):
        for sample in range(20):
            stream = torch.tensor(
                pytorch_tensors[sample + k * 20, :, 0]
                + 1j * pytorch_tensors[sample + k * 20, :, 1]
            ).cuda()
            reference = (
                torch.tensor(kwave_results[sample + k * 20])
                .repeat(iterations, 1, 1)
                .cuda()
            )
            difference, _, __ = difference_to_kwave(stream, reference, None)
            l_infty, indices = difference.reshape(difference.shape[0], -1).topk(1, 1)
            mse = difference.pow(2).mean([1, 2]).sqrt()
            mse_vs_iteration.append(mse.cpu().numpy())
            l_infty_vs_iteration.append(l_infty.cpu().numpy())

    mse_traces = np.array(mse_vs_iteration)
    l_infty_traces = np.array(l_infty_vs_iteration)
    l_infty_traces = l_infty_traces[:, :, 0]

    print("Saving")
    np.savez(
        "results/model_traces.npz", l_infty_traces=l_infty_traces, mse_traces=mse_traces
    )
    return l_infty_traces, mse_traces


def get_gmres_errors(gmres_results, kwave_results):
    print("Getting GMRES error curves")

    if os.path.isfile("results/gmres_traces.npz"):
        npzfile = np.load("results/gmres_traces.npz")
        return npzfile["l_infty_traces_gmres"], npzfile["mse_traces_gmres"]

    print("File not found: generating curves")

    mse_vs_iteration_gmres = []
    l_infty_vs_iteration_gmres = []
    for k in tqdm(range(gmres_results.shape[0])):
        stream = torch.tensor(gmres_results[k])
        reference = torch.tensor(kwave_results[k]).repeat(11, 1, 1)
        difference, _, __ = difference_to_kwave(stream, reference, None)
        l_infty, indices = difference.reshape(difference.shape[0], -1).topk(1, 1)
        mse = difference.pow(2).mean([1, 2]).sqrt()
        mse_vs_iteration_gmres.append(mse.cpu().numpy())
        l_infty_vs_iteration_gmres.append(l_infty.cpu().numpy())

    mse_traces_gmres = np.array(mse_vs_iteration_gmres)
    l_infty_traces_gmres = np.array(l_infty_vs_iteration_gmres)
    l_infty_traces_gmres = l_infty_traces_gmres[:, :, 0]

    print("Saving")
    np.savez(
        "results/gmres_traces.npz",
        l_infty_traces_gmres=l_infty_traces_gmres,
        mse_traces_gmres=mse_traces_gmres,
    )
    return l_infty_traces_gmres, mse_traces_gmres


def normalize_wavefield(wavefield, source_location):
    if len(wavefield.shape) == 2:
        return wavefield / wavefield[source_location[0], source_location[1]]
    elif len(wavefield.shape) == 3:
        return wavefield / wavefield[
            :, source_location[0], source_location[1]
        ].unsqueeze(1).unsqueeze(1)


def show_example(
    sos,
    model_field,
    kwave_field,
    traces,
    traces_name,
    source_location=[82, 48],
    filename=None,
    setticks=True,
):
    sos_map = sos
    kwave_field = normalize_wavefield(np.conj(kwave_field), source_location)
    model_field = normalize_wavefield(model_field, source_location)

    fig, axs = plt.subplots(1, 4, figsize=(12, 2.2), dpi=300)

    raster1 = axs[0].imshow(np.real(kwave_field), vmin=-0.5, vmax=0.5, cmap="seismic")
    axs[0].axis("off")
    axs[0].set_title("Reference")
    fig.colorbar(raster1, ax=axs[0])

    ax = fig.add_axes([0.025, 0.6, 0.25, 0.25])
    raster2 = ax.imshow(sos_map, vmin=1, vmax=2, cmap="inferno")
    ax.axis("off")

    raster3 = axs[1].imshow(np.real(model_field), vmin=-0.5, vmax=0.5, cmap="seismic")
    axs[1].axis("off")
    axs[1].set_title("Prediction")
    fig.colorbar(raster3, ax=axs[1])

    error_field = (kwave_field - model_field)[8:-8, 8:-8]
    error_field = np.pad(error_field, 8)
    raster4 = axs[2].imshow(
        np.log10(np.abs(error_field + 1e-20)), vmin=-4, vmax=-2, cmap="inferno"
    )
    axs[2].axis("off")
    axs[2].set_title("Difference")
    cbar = fig.colorbar(raster4, ax=axs[2])
    cbar.set_ticks(np.log10([0.1, 0.01, 0.001, 0.0001]))
    cbar.set_ticklabels(["10\%", "1\%", "0.1\%", "0.01\%"])

    for trace in traces:
        axs[3].plot(trace["x"],trace["y"], color=trace["color"], label=trace["name"])
    axs[3].set_yscale("log")
    axs[3].set_xscale("log")
    axs[3].set_xlim([1, len(traces[0]["x"])])
    axs[3].set_title(traces_name)
    axs[3].set_xlabel("Iterations")
    axs[3].yaxis.tick_right()
    axs[3].grid(True)
    axs[3].legend()
    if setticks:
        axs[3].set_xticks([1, 10, 100, 1000])
        axs[3].set_xticklabels(["1", "10", "100", "1000"])


def show_example_large(
    sos,
    model_field,
    kwave_field,
    traces,
    traces_name,
    source_location=[82, 48],
    setticks=False,
    filename=None,
):
    sos_map = sos

    kwave_field = normalize_wavefield(np.conj(kwave_field), source_location)
    model_field = normalize_wavefield(model_field, source_location)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

    raster1 = axs[0, 0].imshow(
        np.real(kwave_field), vmin=-0.2, vmax=0.2, cmap="seismic"
    )
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Reference")
    fig.colorbar(raster1, ax=axs[0, 0])

    ax = fig.add_axes([0.117, 0.773, 0.10, 0.10])
    raster2 = ax.imshow(sos_map, vmin=1, vmax=2, cmap="inferno")
    ax.axis("off")

    raster3 = axs[0, 1].imshow(
        np.real(model_field), vmin=-0.2, vmax=0.2, cmap="seismic"
    )
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Prediction")
    # fig.colorbar(raster3, ax=axs[0,1])

    error_field = (kwave_field - model_field)[15:-15, 15:-15]
    error_field = np.pad(error_field, 15)
    raster4 = axs[1, 0].imshow(
        np.log10(np.abs(error_field) + 1e-20), vmin=-4, vmax=-2, cmap="inferno"
    )
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Difference")
    cbar = fig.colorbar(raster4, ax=axs[1, 0])
    cbar.set_ticks(np.log10([0.1, 0.01, 0.001, 0.0001]))
    cbar.set_ticklabels(["10\%", "1\%", "0.1\%", "0.01\%"])

    for trace in traces:
        axs[1, 1].plot(trace["x"],trace["y"], color=trace["color"], label=trace["name"])
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_xlim([1, len(trace)])
    axs[1, 1].set_title(traces_name)
    axs[1, 1].set_xlabel("Iterations")
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    if setticks:
        axs[1, 1].set_xticks([1, 10, 100, 1000])
        axs[1, 1].set_xticklabels(["1", "10", "100", "1000"])
        axs[1, 1].set_yticks([0.0001, 0.001, 0.01, 0.1])
        axs[1, 1].set_ylim([0.00001, 0.01])


def show_example_abs(
    sos,
    model_field,
    kwave_field,
    trace,
    trace_name="Residual RMSE",
    setticks=False,
    filename=None,
):
    sos_map = sos
    kwave_field = np.abs(kwave_field)
    kwave_field /= np.amax(kwave_field)
    model_field = np.abs(model_field)
    model_field /= np.amax(model_field)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

    raster1 = axs[0, 0].imshow(np.real(kwave_field), vmin=0, vmax=0.5, cmap="inferno")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Reference")
    fig.colorbar(raster1, ax=axs[0, 0])

    ax = fig.add_axes([0.117, 0.773, 0.10, 0.10])
    raster2 = ax.imshow(sos_map, vmin=1, vmax=2, cmap="inferno")
    ax.axis("off")

    raster3 = axs[0, 1].imshow(np.real(model_field), vmin=0, vmax=0.5, cmap="inferno")
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Prediction")
    # fig.colorbar(raster3, ax=axs[0,1])

    error_field = (kwave_field - model_field)[15:-15, 15:-15]
    error_field = np.pad(error_field, 15)
    raster4 = axs[1, 0].imshow(
        np.log10(np.abs(error_field) + 1e-20), vmin=-4, vmax=-2, cmap="inferno"
    )
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Difference")
    cbar = fig.colorbar(raster4, ax=axs[1, 0])
    cbar.set_ticks(np.log10([0.1, 0.01, 0.001, 0.0001]))
    cbar.set_ticklabels(["10\%", "1\%", "0.1\%", "0.01\%"])

    axs[1, 1].plot(trace, color="black")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_xlim([1, len(trace)])
    axs[1, 1].set_title(trace_name)
    axs[1, 1].set_xlabel("Iterations")
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].grid(True)
    if setticks:
        axs[1, 1].set_xticks([1, 10, 100, 1000])
        axs[1, 1].set_xticklabels(["1", "10", "100", "1000"])
        axs[1, 1].set_yticks([0.0001, 0.001, 0.01, 0.1])
        axs[1, 1].set_ylim([0.00001, 0.01])


def make_skull_example(evaluator):
    print("----- Running kWave (output not shown)")
    command = [
        "matlab",
        ''' -nodisplay -nosplash -nodesktop -r "run('matlab/skull_example.m'); exit;"''',
    ]
    subprocess.run(command, capture_output=True)

    print("----- Solving with model")
    kwave_solution = loadmat("examples/kwavedata512.mat")["p_kw"]
    matlab_variables = loadmat("examples/problem_setup.mat")
    speedofsound = matlab_variables["sos"].astype(float)
    src_map = 10 * matlab_variables["src"].astype(float)
    sos_map = torch.tensor(speedofsound).unsqueeze(0).unsqueeze(0)
    source = torch.tensor(src_map).unsqueeze(0).float()
    evaluator.set_domain_size(sos_map.shape[-1], source_map=source)
    sos_map_tensor = torch.tensor(sos_map).to("cuda:" + str(evaluator.gpus[0])).float()

    with torch.no_grad():
        output = evaluator.model.forward(
            sos_map_tensor,
            num_iterations=3000,
            return_wavefields=True,
            return_states=False,
        )

    with torch.no_grad():
        losses = [evaluator.model.test_loss_function(x) for x in output["residuals"]]

        pytorch_wavefield = torch.cat(
            [(x[:, 0] + 1j * x[:, 1]).detach().cpu() for x in output["wavefields"]]
        ).cpu()
        kwave_wavefield = torch.tensor(kwave_solution, device=pytorch_wavefield.device)

        max_pt = torch.argmax(torch.abs(kwave_wavefield))
        row, col = max_pt // 512, max_pt - (max_pt // 512) * 512

        kwave_field_norm = normalize_wavefield(
            torch.conj(kwave_wavefield), source_location=[row, col]
        )
        model_field_norm = normalize_wavefield(
            pytorch_wavefield, source_location=[row, col]
        )
        difference = torch.abs(kwave_field_norm.unsqueeze(0) - model_field_norm)[
            :, 15:-15, 15:-15
        ]
        l_infty, indices = difference.reshape(difference.shape[0], -1).topk(1, 1)

    # Store some wavefields
    iterations = np.rint(3000 ** np.linspace(0, 1, 16) - 1).tolist()
    iterations = list(map(int, iterations))

    samples = np.stack([model_field_norm[i].abs().cpu() for i in iterations])

    savemat(
        "examples/pytorch_results.mat",
        {
            "pytorch_wf": pytorch_wavefield[-1].cpu().numpy(),
            "res": np.array(losses),
            "l_infty": np.array(l_infty),
            "samples": samples,
            "iterations": iterations,
        },
    )


def fig_generic(
    solver,
    sos_map,
    path,
    source_location=[82, 48],
    omega=1,
    min_sos=1,
    cfl=0.01,
    roundtrips=60.0,
    mode="normal",
    restart=20,
    max_iter = 1000
):
    assert mode in ["normal", "large"]
    print("Making {}".format(path))
    flag = 0

    # Save data into matfile
    savemat(
        "/tmp/helmholtz_setup.mat",
        {
            "sos_map": sos_map,
            "source_location": source_location,
            "omega": omega,
            "min_sos": min_sos,
            "flag": flag,
            "cfl": cfl,
            "roundtrips": roundtrips,
            "pml_size": solver.hparams.PMLsize,
            "sigma_star": solver.hparams.sigma_max,
            "max_iter": max_iter,
            "restart": restart
        },
    )

    #gmres_matfile = loadmat("/tmp/helmholtz.mat")

    # Solve with kWave
    print("Solving with kWave")
    command = [
        "matlab",
        ''' -nodisplay -nosplash -nodesktop -nojvm -r "run('matlab/solve_with_kwave.m'); exit;"''',
    ]
    subprocess.run(command, capture_output=True)
    matfile = loadmat("/tmp/helmholtz.mat")
    kwave_solution = matfile["p"]
    kwave_wavefield = torch.tensor(kwave_solution)
    kwave_field_norm = normalize_wavefield(
        torch.conj(kwave_wavefield), source_location
    )

    # Solve with gmres
    print("Solving with GMRES")
    #"""
    command = [
        "matlab",
        ''' -nodisplay -nosplash -nodesktop -r "run('matlab/solve_with_gmres.m'); exit;"''',
    ]
    subprocess.run(command, capture_output=True)
    #"""
    
    matfile = loadmat("/tmp/helmholtz.mat")#gmres_matfile# loadmat("/tmp/helmholtz.mat")
    gmres_solution = matfile["p"]
    gmres_error = matfile["rel_error"]

    # Finding GMRES error curve
    kwave_wavefield = torch.tensor(kwave_solution)
    kwave_field_norm = normalize_wavefield(torch.conj(kwave_wavefield), source_location)
    gmres_solutions = torch.tensor(gmres_solution)
    gmres_norm = normalize_wavefield(gmres_solutions, source_location)
    gmres_difference = torch.abs(kwave_field_norm.unsqueeze(0) - gmres_norm)[:, 10:-10, 10:-10]
    l_infty_gmres, indices = gmres_difference.reshape(gmres_difference.shape[0], -1).topk(1, 1)

    # Solving with model
    print("Solving with Neural network")
    sos_map_tensor = (
        torch.tensor(sos_map).unsqueeze(0).unsqueeze(0).to(solver.device)
    ).float()
    with torch.no_grad():
        output = solver.forward(
            sos_map_tensor,
            num_iterations=1000,
            return_wavefields=True,
            return_states=False,
        )

        # Find losses
        losses = [solver.test_loss_function(x) for x in output["residuals"]]

        pytorch_wavefield = torch.cat(
            [x[:, 0] + 1j * x[:, 1] for x in output["wavefields"]]
        )
        kwave_wavefield = torch.tensor(kwave_solution, device=pytorch_wavefield.device)

        kwave_field_norm = normalize_wavefield(
            torch.conj(kwave_wavefield), source_location
        )
        model_field_norm = normalize_wavefield(pytorch_wavefield, source_location)
        difference = torch.abs(kwave_field_norm.unsqueeze(0) - model_field_norm)[
            :, 10:-10, 10:-10
        ]
        l_infty, indices = difference.reshape(difference.shape[0], -1).topk(1, 1)

    traces = [
        {
            "name": "Proposed",
            "x": np.linspace(1,max_iter,max_iter, endpoint=True),
            "y": 100*l_infty.cpu(),
            "color": "black"
        },
        {
            "name": "GMRES",
            "x": np.linspace(1,max_iter, l_infty_gmres.shape[0], endpoint=True),
            "y": 100*l_infty_gmres,
            "color": "darkorange"
        }
    ]

    if mode == "normal":
        show_example(
            sos_map,
            pytorch_wavefield[-1].cpu(),
            kwave_wavefield.cpu(),
            traces,
            traces_name = "$\ell_\infty$ error %",
            source_location=source_location,
        )
    elif mode == "large":
        show_example_large(
            sos_map,
            pytorch_wavefield[-1].cpu(),
            kwave_wavefield.cpu(),
            traces,
            traces_name = "$\ell_\infty$ error %",
            source_location=source_location,
        )

    plt.savefig(path + ".pgf")
