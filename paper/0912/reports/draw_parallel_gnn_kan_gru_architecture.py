#!/usr/bin/env python3
"""Draw the reviewed parallel GNN-KAN-GRU forecasting architecture."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = [
    "Noto Sans SC",
    "Microsoft YaHei",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
plt.rcParams["font.size"] = 6.5


PALETTE = {
    "blue": "#0F4D92",
    "blue_fill": "#E6F0FA",
    "teal": "#2C7F7B",
    "teal_fill": "#DDF1EE",
    "violet": "#7651A8",
    "violet_fill": "#EEE7F7",
    "orange": "#C56B16",
    "orange_fill": "#FBEAD7",
    "slate": "#52687A",
    "slate_fill": "#E9EEF2",
    "fusion": "#31435B",
    "decoder": "#2D6E3F",
    "decoder_fill": "#E4F2E6",
    "text": "#272727",
    "muted": "#666666",
    "line": "#6B7785",
}


def rounded_box(ax, x, y, w, h, text, face, edge, *, fontsize=6.2,
                color=None, linewidth=1.15, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color or PALETTE["text"],
        fontweight=weight,
        linespacing=1.25,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color, *, lw=1.3, style="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        linestyle=style,
        color=color,
        shrinkA=1,
        shrinkB=1,
        connectionstyle="arc3,rad=0",
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    base = out_dir / "parallel_gnn_kan_gru_architecture_v1"

    width_mm, height_mm = 183, 122
    fig, ax = plt.subplots(figsize=(7.2047, 4.8031))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "并行 GNN–KAN–GRU 五分钟 Thv 轨迹预测架构",
        x=0.5,
        y=0.975,
        fontsize=10,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        0.5,
        0.93,
        "静态拓扑与 PCMCI 用于配置滞后边；各编码分支在前向计算中并行",
        ha="center",
        va="center",
        fontsize=6.4,
        color=PALETTE["muted"],
    )

    rows = [0.78, 0.635, 0.49, 0.345, 0.20]
    h = 0.095
    input_x, input_w = 0.025, 0.205
    branch_x, branch_w = 0.29, 0.235
    fusion_x, fusion_y, fusion_w, fusion_h = 0.665, 0.39, 0.12, 0.22
    decoder_x, decoder_y, decoder_w, decoder_h = 0.84, 0.36, 0.145, 0.28

    inputs = [
        ("Raw54 历史窗口\n过程量与阀门双通道", PALETTE["blue_fill"], PALETTE["blue"]),
        ("节点历史 + 滞后图\nedge_index / edge_lag / edge_attr", PALETTE["teal_fill"], PALETTE["teal"]),
        ("三支路状态\n阀门、温度、压力", PALETTE["violet_fill"], PALETTE["violet"]),
        ("A管 + EC-V2\nCOOLDOWN + FC-V1", PALETTE["orange_fill"], PALETTE["orange"]),
        ("未来控制计划（可选）\n或保持/场景假设", PALETTE["slate_fill"], PALETTE["slate"]),
    ]
    branches = [
        ("GRU 全局时序编码器\nh_global", PALETTE["blue_fill"], PALETTE["blue"]),
        ("Lag-GNN 物理分支\n更新全图节点，读取 h_Thv^GNN", PALETTE["teal_fill"], PALETTE["teal"]),
        ("KAN 分流边函数\nαT(t), αP(t), z_branch", PALETTE["violet_fill"], PALETTE["violet"]),
        ("KAN 虚拟汇合 H_mix\n按 PCMCI 时延对齐输入", PALETTE["orange_fill"], PALETTE["orange"]),
        ("控制序列编码器\nh_control", PALETTE["slate_fill"], PALETTE["slate"]),
    ]

    for y, inp, branch in zip(rows, inputs, branches):
        rounded_box(ax, input_x, y - h / 2, input_w, h, *inp)
        rounded_box(ax, branch_x, y - h / 2, branch_w, h, *branch)
        arrow(ax, (input_x + input_w, y), (branch_x, y), branch[2])

    rounded_box(
        ax,
        fusion_x,
        fusion_y,
        fusion_w,
        fusion_h,
        "门控融合\nconcat + gate\nz(t)",
        PALETTE["fusion"],
        PALETTE["fusion"],
        fontsize=7.0,
        color="white",
        weight="bold",
    )
    rounded_box(
        ax,
        decoder_x,
        decoder_y,
        decoder_w,
        decoder_h,
        "多步轨迹解码器\nGRU / MLP\n\n输出 ΔThv(t+1:t+30)\n与当前 Thv(t) 累加\n\n得到未来 5 min\nThv 温度轨迹",
        PALETTE["decoder_fill"],
        PALETTE["decoder"],
        fontsize=6.6,
        weight="bold",
    )

    fusion_targets = [0.575, 0.535, 0.50, 0.465, 0.425]
    branch_colors = [
        PALETTE["blue"], PALETTE["teal"], PALETTE["violet"],
        PALETTE["orange"], PALETTE["slate"],
    ]
    for y, y_target, color in zip(rows, fusion_targets, branch_colors):
        arrow(ax, (branch_x + branch_w, y), (fusion_x, y_target), color, lw=1.25)

    arrow(
        ax,
        (fusion_x + fusion_w, fusion_y + fusion_h / 2),
        (decoder_x, decoder_y + decoder_h / 2),
        PALETTE["fusion"],
        lw=2.0,
    )

    ax.text(
        0.025,
        0.085,
        "时延读取：边 i→j 若 lag=τ，则构造 j(t) 时读取 i(t−τ)；difference 边读取 Δi，level 边读取 i。",
        ha="left",
        va="center",
        fontsize=5.8,
        color=PALETTE["muted"],
    )
    ax.text(
        0.025,
        0.035,
        "PCMCI 是离线构图与对齐步骤，不是额外的串行神经网络层；未知未来阀门需由 MPC 计划或场景假设提供。",
        ha="left",
        va="center",
        fontsize=5.8,
        color=PALETTE["muted"],
    )

    fig.subplots_adjust(left=0.015, right=0.99, top=0.92, bottom=0.02)
    fig.canvas.draw()

    skill_scripts = Path(r"C:\Users\Administrator\.codex\skills\nature-figure\scripts")
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["architecture"],
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        strict=True,
    )

    fig.savefig(str(base) + ".svg", bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
