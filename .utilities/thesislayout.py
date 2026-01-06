from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes


def get_figure(
    figsize=(1.7, 0.85),
    facecolor=None,
    subfigure=False,
):
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=8,
        pad=1.5,
    )
    ax.subfigure: bool = subfigure
    if ax.subfigure:
        labelsize = 7
        ax.xaxis.label.set_size(7)
        ax.yaxis.label.set_size(7)
        ax.tick_params(labelsize=labelsize)
    else:
        labelsize = 8

    # Remove frame
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Simplify ticks
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=labelsize,
        color="k",
        labelcolor="k",
    )
    return fig, ax


def theory_layout(
    fig: Figure,
    ax: Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    padding: Optional[tuple[float, float]] = None,
    base_padding: Optional[tuple[float, float]] = None,
):
    if xlabel is None:
        xlabel = "$x$"

    if ylabel is None:
        ylabel = "$y$"

    if padding is None:
        padding = (0.2, 0.2)

    if base_padding is None:
        base_padding = (0.08, 0.08)

    ms: float = 6.0
    fontsize: float = 8
    if ax.subfigure:
        ms: float = 4.0
        fontsize: float = 7

    # get limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Draw arrow heads
    ax.plot(x1, y0, ">", color="k", ms=ms, clip_on=False)
    ax.plot(x0, y1, "^", color="k", ms=ms, clip_on=False)

    # get config stuff
    w0, h0 = fig.get_size_inches()
    px, py = padding
    bpx, bpy = base_padding
    w1 = w0 - (px + 2 * bpx)
    h1 = h0 - (py + 2 * bpy)
    dx = np.abs(x1 - x0)
    dy = np.abs(y1 - y0)

    # make layout
    ax.set_position(
        (
            (bpx + px) / w0,
            (bpy + py) / h0,
            w1 / w0,
            h1 / h0,
        ),
    )

    # labels
    ax.text(
        # x = x1 + bpx / w1 * dx, #label on the right
        # ha="right",
        x=x0 + dx / 2,  # label in the middle
        ha="center",
        y=y0 - (py + bpy) / h1 * dy,
        s=xlabel,
        va="bottom",
        fontsize=fontsize,
    )
    ax.text(
        x0 - (bpx + px) / w1 * dx,
        y0 + dy / 2,
        ylabel,
        rotation=90,
        ha="left",
        va="center",
        fontsize=fontsize,
    )

    # set limits
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))

    # save figure
    if title is not None:
        fig.savefig(f"{title}.pgf")
        fig.savefig(f"{title}.pdf")
