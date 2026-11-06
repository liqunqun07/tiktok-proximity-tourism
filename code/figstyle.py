"""Shared journal-style figure settings (serif type, restrained black/grey palette)."""
from __future__ import annotations

import os

import matplotlib as mpl
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# fixed PDF CreationDate so re-running the pipeline gives byte-identical figures
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

INK = "#111111"
GREY = "#6b6b6b"
LIGHT = "#f1f1f1"
MID = "#d9d9d9"


def apply() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def box(ax, x, y, w, h, lines=(), *, fc="white", ec=INK, lw=0.8, ls="-", size=7.6, bold_first=True,
        align="center", rounding=0.6, pad_x=1.0, italic_second=False, zorder=2, head_size=None):
    """Rounded box in data coordinates; ``lines`` are drawn top to bottom (first line bold)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=zorder))
    n = len(lines)
    if not n:
        return
    gap = h / (n + 0.6)
    for i, t in enumerate(lines):
        ty = y + h - gap * (i + 0.8)
        tx = x + w / 2 if align == "center" else x + pad_x
        ax.text(tx, ty, t, ha=align, va="center", fontsize=(head_size or size) if (bold_first and i == 0) else size, color=INK, zorder=zorder + 1,
                fontweight="bold" if (bold_first and i == 0) else "normal",
                fontstyle="italic" if (italic_second and i == 1) else "normal")


def arrow(ax, p, q, *, lw=0.8, color=INK, style="-|>", ls="-", rad=0.0, ms=7, zorder=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=zorder, shrinkA=0, shrinkB=0))


def tag(ax, x, y, text, size=7.2):
    """Small white label with thin border (hypothesis tags)."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=INK, lw=0.6))
