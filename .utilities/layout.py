from typing import Optional
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure, Axes


def get_figure(
    figsize=(1.7, 0.85),
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis="both", direction="out", length=3, labelsize=8, pad=1.5)

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
    padding: tuple[float, float] = (0.35, 0.15),
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
    bpx, bpy = base_padding
    px, py = padding

    # make layout
    ax.set_position(
        (
            (bpx + px) / w0,
            (bpy + py) / h0,
            (w0 - 2 * bpx - px) / w0,
            (h0 - 2 * bpy - py) / h0,
        ),
    )

    # labels
    ax.text(
        x1 + bpx / (w0 - (px + 2 * bpx)) * np.abs(x1 - x0),
        y0 - (py + bpy) / (h0 - (py + 2 * bpy)) * np.abs(y1 - y0),
        xlabel,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    ax.set_ylabel(ylabel)
    ax.yaxis.set_label_coords(
        (-px * 0.5) / (w0 - (px + 2 * bpx)), 0.5
    )  # fixed relative to axes box

    # set limits
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))

    # save figure
    if title is not None:
        fig.savefig(f"{title}.pgf")
        fig.savefig(f"{title}.pdf")
