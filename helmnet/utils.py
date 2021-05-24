import json
import os
from matplotlib import pyplot as plt
import numpy as np


def load_settings(jsonpath: str, add_full_path=True):
    """Loads a `settings.json` file and adds the folder path to its
    fields

    Args:
        folder (str): folder path
        add_full_path (bool, optional): If true, adds the folder path to its
            fields. Defaults to True.
    """

    with open(jsonpath) as json_file:
        settings = json.load(json_file)
    if add_full_path:
        settings["path"] = jsonpath
        settings["name"] = os.path.splitext(os.path.basename(jsonpath))[0]
    return settings


def show_wavefield(wf, component="real", crange=0, colorbar=True, colormap="seismic"):
    """Helper function to plot a wavefield

    Args:
        wf (np.array): Wavefield to be shown. Must have the last dimension of
            size 2, representing real and imaginary part.
        component (str, optional): Which component to plot: can be "real" or "imag".
            Defaults to "real".
        crange (float, optional): The colormap will display values in (-crange, crange).
            If 0, it is given by the maximum absolute amplitude. Defaults to 0.
        colorbar (bool, optional): If a colorbar has to be used. Defaults to True.
        colormap (str, optional): What colormap to use. Defaults to 'seismic'.
    """
    if crange == 0:
        crange = np.sqrt(np.max(np.sum(wf[0] ** 2 + wf[1] ** 2))) / 20
    elif crange < 0:
        raise ValueError("The range must be a positive number")

    if component == "real":
        _show_image(
            wf[0], vmin=-crange, vmax=crange, colorbar=colorbar, colormap=colormap
        )
    elif component == "imag":
        _show_image(
            wf[1], vmin=-crange, vmax=crange, colorbar=colorbar, colormap=colormap
        )
    else:
        raise ValueError('The component field can be either "real" or "imag".')


def log_wavefield(wavefield, logger, windowname="Wavefield"):
    """Logs a wavefield map image to tensorboard."""
    wavefield = wavefield.cpu()
    fig = plt.figure(figsize=(6, 3))
    plt.title(windowname)
    plt.subplot(1, 2, 1)
    show_wavefield(wavefield, component="real", crange=1)
    plt.subplot(1, 2, 2)
    show_wavefield(wavefield, component="imag", crange=1)
    plt.tight_layout()
    logger.add_figure(windowname, fig, 0)
    plt.close()


def _show_image(image, vmin=None, vmax=None, colorbar=True, colormap="hot"):
    """Helper function to show an image with colorbar and
    custom colormap extrema.

    Args:
        image ([type]): Image to be shown
        vmin ([type], optional): Custom `vmin`. If `None` this value
            is the minimum of the image. Defaults to None.
        vmax ([type], optional): Custom `vmax`. If `None` this value
            is the maximum of the image. Defaults to None.
        colorbar (bool, optional): If a colorbar has to be used. Defaults
            to True.
        colormap (str, optional): What colormap to use. Defaults to 'hot'.
    """
    if vmin is None:
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap=colormap, aspect="equal")
    if colorbar:
        plt.colorbar()


# A function to rasterize components of a matplotlib figure while keeping
# axes, labels, etc as vector components
# https://brushingupscience.wordpress.com/2017/05/09/vector-and-raster-in-one-with-matplotlib/
from inspect import getmembers, isclass
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None, savefig_kw={}):
    """Save a figure with raster and vector components
    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.

    Code from:
    https://gist.github.com/hugke729/78655b82b885cde79e270f1c30da0b5f

    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig
    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized
    Note: does not work correctly with round=True in Basemap
    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line
    >>> import matplotlib.pyplot as plt
    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ["QuadMesh", "Contour", "collections"]
        rasterize_list = []

        print(
            """
        No rasterize_list specified, so the following objects will
        be rasterized: """
        )
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print("\n".join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = isinstance(item, matplotlib.contour.QuadContourSet) or isinstance(
            item, matplotlib.tri.TriContourSet
        )

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw["dpi"] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)
