from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes


def get_figure(
    figsize=(1.7, 0.85),
    facecolor=None,
):
    fig, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=8,
        pad=1.5,
    )

    # Remove frame
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Simplify ticks
    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        labelsize=8,
        color="k",
        labelcolor="k",
    )
    return fig, ax


def theory_layout(
    fig: Figure,
    ax: Axes,
    title: Optional[str] = None,
    xlabel: str = "$x$",
    ylabel: str = "$y$",
    padding: tuple[float, float] = (0.3, 0.15),
    base_padding: tuple[float, float] = (0.08, 0.08),
):

    # get limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    # Draw arrow heads
    ax.plot(x1, y0, ">", color="k", clip_on=False)
    ax.plot(x0, y1, "^", color="k", clip_on=False)

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
        x1 + bpx / w1 * dx,
        y0 - (py + bpy) / h1 * dy,
        xlabel,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.text(
        x0 - (bpx + px) / w1 * dx,
        y0 + dy / 2,
        ylabel,
        rotation=90,
        ha="left",
        va="center",
        fontsize=8,
    )

    # set limits
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))

    # save figure
    if title is not None:
        fig.savefig(f"{title}.pgf")
        fig.savefig(f"{title}.pdf")
