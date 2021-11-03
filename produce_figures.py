import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from torchvision.utils import make_grid

from evaluate import Evaluation
from helmnet.support_functions import (get_model_errors, get_gmres_errors, last_frame_difference, fig_generic,
                                       make_skull_example, show_example_abs)

SETTINGS = {
    "gmres_results": "results/gmres_results.mat",
    "kwave_results": "results/kwave_results.mat",
    "model_checkpoint": "checkpoints/trained_weights.ckpt",
    "testset": "datasets/splitted_96/testset.ph",
    "gpu": [0],
}


def load_kwave_and_gmres():
    if not os.path.isfile(SETTINGS["kwave_results"]):
        raise FileNotFoundError(
            "Can't find the k-Wave results in {}. Have you run 'matlab/parallel_kwave_solver.m'?".format(
                SETTINGS["kwave_results"]
            )
        )
    if not os.path.isfile(SETTINGS["gmres_results"]):
        raise FileNotFoundError(
            "Can't find the GMRES results in {}. Have you run 'matlab/parallel_sectral_gmres_solver.m'?".format(
                SETTINGS["gmres_results"]
            )
        )
    # Load data
    print("Loading k-Wave and GMRES results... ", end="")
    matfile = loadmat(SETTINGS["kwave_results"])
    kwave_results = matfile["P"]
    matfile = loadmat(SETTINGS["gmres_results"])
    gmres_results = matfile["P"]
    gmres_residuals = (
            matfile["residuals"] / gmres_results.shape[-1]
    )  # To mimick RMSE used in network
    print("done!")

    gmres_tensors = np.moveaxis(
        np.stack([gmres_results.real, gmres_results.imag]), 0, 2
    )
    return kwave_results, gmres_results, gmres_residuals, gmres_tensors


def load_model_results():
    path = "results/evolution_of_wavefields_on_test_set.npy"
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Can't find the model results on the testset. Have you run 'python evaluate.py'?"
        )

    print("Loading model results, this may take some time... ", end="")
    pytorch_tensors = np.load("results/evolution_of_wavefields_on_test_set.npy")
    traces_file = "results/evolution_of_model_RMSE_on_test_set.npy"
    traces = np.load(traces_file)
    print("done!")

    return pytorch_tensors, traces


def fig_samples_from_testset(evaluator, savepath="images/example_skulls"):
    print("Saving examples from testset in {}".format(savepath))
    some_sos_maps = make_grid([evaluator.testset[i] for i in range(8 * 8)], nrow=8)
    plt.figure(figsize=(8, 8), dpi=300)
    plt.imshow(some_sos_maps.cpu().numpy()[0], vmin=1.0, vmax=2.0, cmap="inferno")
    plt.colorbar(fraction=0.02, pad=0.02)
    plt.axis("off")
    plt.savefig(savepath + ".png")


def fig_error_vs_residual(
        traces,
        l_infty_traces,
        path="images/error_vs_residual",
        iterations=1000,
        lines_color="darkgray",
        lines_alpha=0.1,
        mean_color="black",
        xscale="log",
        yscale="log",
        dpi=100,
):
    print("Making Error vs Residual figure")

    plt.figure(dpi=dpi)

    toraster = plt.plot(
        traces.T, 100 * l_infty_traces.T, color=lines_color, alpha=lines_alpha
    )
    mean_residual = np.mean(traces, 0)
    mean_error = np.mean(100 * l_infty_traces, 0)
    plt.plot(mean_residual, mean_error, color=mean_color, linestyle="--", label="Mean")
    median_residual = np.median(traces, 0)
    median_error = np.median(100 * l_infty_traces, 0)
    plt.plot(median_residual, median_error, color=mean_color, label="Median")

    plt.yscale(yscale)
    plt.xscale(xscale)
    plt.xlabel("Residual magnitude")
    plt.ylabel("$\ell_\infty$ error (percent)")
    plt.ylim([0.1, 100])
    plt.xlim([1e-5, 1e-1])
    plt.grid()
    plt.legend()
    plt.savefig(path + ".png")


def fig_residual_and_error_traces(
        traces,
        l_infty_traces,
        gmres_traces,
        l_infty_traces_gmres,
        path="images/residual_and_l_inf",
        dpi=100,
        iterations=1000,
        lines_alpha=0.05,
        xscale="linear",
        yscale="log",
):
    gmres_x = np.linspace(1, 1000, gmres_traces.shape[1])

    w, h = plt.figaspect(1 / 3.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, h), dpi=dpi)

    toraster1 = ax1.plot(gmres_x, gmres_traces.T, color="orange", alpha=lines_alpha)
    ax1.plot(gmres_x, np.mean(gmres_traces, 0), color="darkorange", linestyle="--")
    ax1.plot(gmres_x, np.median(gmres_traces, 0), color="darkorange", label="GMRES")

    toraster2 = ax1.plot(traces.T, color="darkgray", alpha=lines_alpha)
    ax1.plot(np.mean(traces, 0), color="black", linestyle="--")
    ax1.plot(np.median(traces, 0), color="black", label="Learned")
    ax1.set_yscale(yscale)
    ax1.set_xscale(xscale)
    ax1.set_title("Residual magnitude")
    ax1.set_xlabel("Number of iterations")
    ax1.set_ylim([0.00001, 0.1])
    ax1.set_xlim([1, 1000])
    ax1.grid()
    ax1.legend()

    x = np.linspace(1, 1001, 1000)
    toraster3 = ax2.plot(x, 100 * l_infty_traces.T, color="darkgray", alpha=lines_alpha)
    ax2.plot(x, np.mean(100 * l_infty_traces, 0), color="black", linestyle="--")
    ax2.plot(x, np.median(100 * l_infty_traces, 0), color="black", label="Learned")
    x = np.linspace(1, 1001, 11)
    toraster4 = ax2.plot(
        x, 100 * l_infty_traces_gmres.T, color="orange", alpha=lines_alpha
    )
    ax2.plot(
        x, np.mean(100 * l_infty_traces_gmres, 0), color="darkgoldenrod", linestyle="--"
    )
    ax2.plot(
        x,
        np.median(100 * l_infty_traces_gmres, 0),
        color="darkgoldenrod",
        label="GMRES",
    )

    ax2.set_yscale(yscale)
    ax1.set_xscale(xscale)
    ax2.set_title("Error $\ell_\infty$ (percent)")
    ax2.set_xlabel("Number of iterations")
    ax2.set_yticks([0.01, 0.1, 1, 10, 100])
    ax2.set_yticklabels(["0.01", "0.1", "1", "10", "100"])
    ax2.set_ylim([0.1, 100])
    ax2.set_xlim([1, iterations])
    ax2.grid()

    plt.savefig(path + ".png")


def histograms(l_infty_pytorch, mse_pytorch, l_infty_gmres, mse_gmres, filename=None):
    kwargs = dict(histtype="stepfilled", alpha=0.5, bins=50, ec="k")

    x_ticks = np.array([0.0001, 0.001, 0.01, 0.1, 1])
    x_ticks_location = np.log10(x_ticks)
    x_thicks_labels = 100 * x_ticks

    fig, axes = plt.subplots(1, 3, figsize=(12, 3), dpi=300)

    axes[0].hist(
        np.log10(l_infty_pytorch.cpu()), **kwargs, color="black", label="Learned"
    )
    axes[0].hist(np.log10(l_infty_gmres.cpu()), **kwargs, color="orange", label="GMRES")
    axes[0].set_xticks(x_ticks_location)
    axes[0].set_xticklabels(x_thicks_labels)
    axes[0].set_xlim([-4, 0])
    axes[0].set_xlabel("$\ell_\infty$ error (\%)")
    axes[0].set_ylabel("Number of")
    axes[0].legend()

    axes[1].hist(np.log10(mse_pytorch.cpu()), **kwargs, color="black")
    axes[1].hist(np.log10(mse_gmres.cpu()), **kwargs, color="orange")
    axes[1].set_xticks(x_ticks_location)
    axes[1].set_xticklabels(x_thicks_labels)
    axes[1].set_xlim([-4, 0])
    axes[1].set_xlabel("RMSE error (x 100)")
    axes[1].set_ylabel("Number of")

    color = "black"
    axes[2].boxplot(
        np.log10(l_infty_pytorch.cpu()),
        positions=(0.85,),
        patch_artist=True,
        boxprops=dict(facecolor="white", color=color),
        flierprops=dict(markerfacecolor=color, marker=".", markersize=1),
        medianprops=dict(color=color),
    )
    color = "darkorange"
    axes[2].boxplot(
        np.log10(l_infty_gmres.cpu()),
        positions=(1.15,),
        patch_artist=True,
        boxprops=dict(facecolor="white", color=color),
        flierprops=dict(markerfacecolor=color, marker=".", markersize=1),
        medianprops=dict(color=color),
    )

    color = "black"
    axes[2].boxplot(
        np.log10(mse_pytorch.cpu()),
        positions=(1.85,),
        patch_artist=True,
        boxprops=dict(facecolor="white", color=color),
        flierprops=dict(markerfacecolor=color, marker=".", markersize=1),
        medianprops=dict(color=color),
    )
    color = "darkorange"
    axes[2].boxplot(
        np.log10(mse_gmres.cpu()),
        positions=(2.15,),
        patch_artist=True,
        boxprops=dict(facecolor="white", color=color),
        flierprops=dict(markerfacecolor=color, marker=".", markersize=1),
        medianprops=dict(color=color),
    )

    axes[2].set_xlim([0.7, 2.3])
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels(["$\ell_\infty (\%)$", "RMSE (x100)"])
    axes[2].set_yticks(x_ticks_location)
    axes[2].set_yticklabels(x_thicks_labels)
    axes[2].yaxis.tick_right()
    axes[2].set_title("$\ell_\infty$  and RMSE errors")

    if filename is not None:
        plt.savefig(filename)


def fig_skull_error_histograms_and_boxplot(
        pytorch_tensors,
        gmres_tensors,
        kwave_results,
        path="images/distribution_errors_global",
):
    l_infty_pytorch, mse_pytorch = last_frame_difference(
        pytorch_tensors[:, :-1], kwave_results
    )
    l_infty_gmres, mse_gmres = last_frame_difference(
        gmres_tensors[:, :-1], kwave_results
    )

    histograms(
        l_infty_pytorch,
        mse_pytorch,
        l_infty_gmres,
        mse_gmres,
        filename=path + ".png",
    )


def fig_example(
        evaluator,
        sos_map,
        path,
        source_location=(82, 48),
        omega=1,
        min_sos=1,
        cfl=0.01,
        roundtrips=60.0,
        mode="normal",
        restart=10,
        max_iter=1000,
):
    solver = evaluator.model
    fig_generic(
        solver,
        sos_map,
        path,
        source_location,
        omega,
        min_sos,
        cfl,
        roundtrips,
        mode,
        restart,
        max_iter
    )


def fig_skull_example(evaluator, path="images/skull_example"):
    if not os.path.isfile("examples/kwavedata512.mat"):
        print("Data for skull example not found, I'll generate it.")
        make_skull_example(evaluator)

    sos_map = loadmat("examples/problem_setup.mat")["sos"]
    kwave_wavefield = loadmat("examples/kwavedata512.mat")["p_kw"]
    pytorch_wavefield = loadmat("examples/pytorch_results.mat")["pytorch_wf"]
    l_infty = loadmat("examples/pytorch_results.mat")["l_infty"]

    show_example_abs(
        sos_map,
        pytorch_wavefield,
        kwave_wavefield,
        100 * l_infty,
        trace_name="$\ell_\infty$ error \%",
    )
    plt.savefig(path + ".png")
    plt.close()

    # Sample iterations
    samples = loadmat("examples/pytorch_results.mat")["samples"]
    iterations = loadmat("examples/pytorch_results.mat")["iterations"][0]

    fig, axs = plt.subplots(4, 4, figsize=(18, 18), dpi=300)
    counter = 0
    for r in range(4):
        for c in range(4):
            plotnum = r * 4 + c
            axs[r, c].imshow(samples[counter], cmap="inferno")
            print(plotnum, len(iterations))
            axs[r, c].set_title("Iteration {}".format(iterations[plotnum] + 1))
            axs[r, c].axis("off")
            counter += 1
    plt.savefig(path + "_evolution.png")


if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    # Load model
    evaluator = Evaluation(
        path=SETTINGS["model_checkpoint"],
        testset=SETTINGS["testset"],
        gpus=SETTINGS["gpu"],
    )
    evaluator.move_model_to_gpu()

    # ----------------------------------------------------------------

    # Load GMRES and kWave results
    kwave_results, gmres_results, gmres_traces, gmres_tensors = load_kwave_and_gmres()

    # Load model results on testset
    pytorch_tensors, traces = load_model_results()

    # Load model
    evaluator = Evaluation(
        path=SETTINGS["model_checkpoint"],
        testset=SETTINGS["testset"],
        gpus=SETTINGS["gpu"],
    )
    evaluator.move_model_to_gpu()

    # ----------------------------------------------------------------

    # Save examples of speed of sound maps from the testset()
    fig_samples_from_testset(evaluator)

    # Evaluate error curves
    l_infty_traces, mse_traces = get_model_errors(pytorch_tensors, kwave_results)
    l_infty_traces_gmres, mse_traces_gmres = get_gmres_errors(
        gmres_results, kwave_results
    )

    # Residual vs error figure
    fig_error_vs_residual(traces, l_infty_traces)
    fig_residual_and_error_traces(
        traces, l_infty_traces, gmres_traces, l_infty_traces_gmres
    )

    # Histograms and boxplots
    fig_skull_error_histograms_and_boxplot(
        pytorch_tensors, gmres_tensors, kwave_results
    )

    # Make examples
    print(
        "--- Example images ---\nEach example may take a while to compute as it runs an accurate kWave simulation (cfl=0.01, roundtrips=60)"
    )
    fig_example(
        evaluator, (evaluator.testset[0]).clone().numpy()[0], path="images/example_0"
    )
    fig_example(
        evaluator, (evaluator.testset[1]).clone().numpy()[0], path="images/example_1"
    )
    fig_example(
        evaluator, (evaluator.testset[2]).clone().numpy()[0], path="images/example_2"
    )
    fig_example(
        evaluator, (evaluator.testset[3]).clone().numpy()[0], path="images/example_3"
    )
    fig_example(
        evaluator, (evaluator.testset[4]).clone().numpy()[0], path="images/example_4"
    )
    fig_example(
        evaluator,
        (evaluator.testset[864]).clone().numpy()[0],
        path="images/worst_example",
    )

    # Rectangle example
    sos_map = (evaluator.testset[0] * 0 + 1).numpy()[0]
    sos_map[20:60, 20:-20] = 2.0
    fig_example(evaluator, sos_map, path="images/rectangle", cfl=0.01, roundtrips=60)

    # Large example
    source_location = [450, 256]
    sos_maps = [evaluator.testset[n] for n in range(25)]
    sos_map = make_grid(sos_maps, nrow=5, padding=0)[0].numpy()
    sos_map[400:, 200:300] = 1.0  # Remove one
    sos_map = np.pad(sos_map, 16, mode="edge")  # Pad to 512x512
    evaluator.set_domain_size(sos_map.shape[-1], source_location=source_location)

    fig_example(
        evaluator,
        sos_map,
        "images/patches",
        source_location=source_location,
        cfl=0.1,
        roundtrips=100,
        mode="large",
        restart=25,
    )

    # Skull example
    fig_skull_example(evaluator)
