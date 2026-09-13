"""Build the operator-review draft of the HTC8300 static physical prior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
plt.rcParams["font.size"] = 7


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "model_code" else SCRIPT_DIR
CATALOG = PROJECT / "data_cache" / "raw54_signal_catalog.csv"
STANDALONE_FIGURE_ONLY = not CATALOG.exists()
GRAPH_DIR = PROJECT / "graphs"
FIGURE_DIR = SCRIPT_DIR if STANDALONE_FIGURE_ONLY else PROJECT / "figures"
DOC_DIR = PROJECT / "docs"

AUDIT_DIR = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(AUDIT_DIR))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment = None


def relation(
    source: str,
    destination: str,
    mechanism: str,
    confidence: str,
    lag_min: int,
    lag_max: int,
    expected_sign: str,
    provenance: str,
    relation_group: str,
) -> dict[str, object]:
    return {
        "source": source,
        "destination": destination,
        "mechanism": mechanism,
        "confidence": confidence,
        "lag_min": lag_min,
        "lag_max": lag_max,
        "lag_unit": "10-second step",
        "lag_status": "candidate_window_not_pcmci_result",
        "expected_sign": expected_sign,
        "provenance": provenance,
        "relation_group": relation_group,
    }


def build_relations() -> list[dict[str, object]]:
    old = "0908_supplied_schematic"
    confirmed = "operator_confirmed_2026-09-12"
    persistence = "state_persistence_proposed_2026-09-12"
    r: list[dict[str, object]] = []

    def add(*args: object) -> None:
        r.append(relation(*args))

    # Retained cross-variable relations from the 0908 prior.
    add("CV8300", "TE8310", "G1 inlet valve changes premix temperature", "high", 1, 6, "context_dependent", old, "retained_0908")
    add("CV8300", "PT8310", "G1 inlet valve changes premix pressure", "high", 1, 6, "context_dependent", old, "retained_0908")
    add("CV8313", "TE8310", "G2 inlet valve changes premix temperature", "high", 1, 6, "context_dependent", old, "retained_0908")
    add("CV8313", "PT8310", "G2 inlet valve changes premix pressure", "high", 1, 6, "context_dependent", old, "retained_0908")
    add("CV8310", "FT8351", "G main-path valve changes downstream helium flow", "high", 1, 6, "usually_positive", old, "retained_0908")
    add("CV8310", "TE8351", "G stream admission changes post-mix temperature", "high", 1, 12, "context_dependent", old, "retained_0908")
    add("CV8310", "PT8351", "G stream admission changes post-mix pressure", "high", 1, 8, "context_dependent", old, "retained_0908")
    add("CV8351", "FT8351", "4.6 K A-line valve changes mixed-line flow", "high", 1, 6, "usually_positive", old, "retained_0908")
    add("CV8351", "TE8351", "4.6 K A-line valve changes post-mix temperature", "high", 1, 12, "usually_negative", old, "retained_0908")
    add("CV8351", "PT8351", "A-line valve changes post-mix pressure", "high", 1, 8, "context_dependent", old, "retained_0908")
    add("TE8310", "TE8351", "premix thermal state is advected to the post-mix sensor", "high", 1, 18, "usually_positive", old, "retained_0908")
    add("PT8310", "PT8351", "upstream pressure propagates along the admitted G path", "high", 1, 12, "usually_positive", old, "retained_0908")
    add("PT8310", "FT8351", "upstream pressure provides a flow-driving condition", "medium", 1, 12, "context_dependent", old, "retained_0908")
    add("TE8351", "TE8352", "serial pipe transport and heat leak", "high", 1, 12, "usually_positive", old, "retained_0908")
    add("PT8351", "PT8352", "serial pressure propagation", "high", 1, 8, "usually_positive", old, "retained_0908")
    add("FT8351", "TE8352", "mass flow modulates downstream cooling and residence time", "high", 1, 12, "context_dependent", old, "retained_0908")
    add("FT8351", "PT8352", "mass flow and pressure loss are coupled", "medium", 1, 8, "context_dependent", old, "retained_0908")
    add("TE8352", "TE8353", "serial pipe transport to the module-side line", "high", 1, 12, "usually_positive", old, "retained_0908")
    add("PT8352", "TE8353", "pressure affects helium state and cooling capacity", "medium", 1, 12, "context_dependent", old, "retained_0908")
    add("FT8351", "TE8353", "delivered flow changes the downstream inlet cooling trend", "high", 1, 18, "context_dependent", old, "retained_0908")

    # New operator-confirmed topology. Lag ranges remain search windows.
    add("CV8312", "TE8330", "branch valve changes the local branch temperature", "high", 1, 12, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8312", "PT8330", "branch valve changes the local branch pressure", "high", 1, 8, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8330", "TE8330", "local valve changes the local branch temperature", "high", 1, 12, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8330", "PT8330", "local valve changes the local branch pressure", "high", 1, 8, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8311", "TE8350", "branch valve changes the local branch temperature", "high", 1, 12, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8311", "PT8350", "branch valve changes the local branch pressure", "high", 1, 8, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8350", "TE8350", "local valve changes the local branch temperature", "high", 1, 12, "context_dependent", confirmed, "new_operator_confirmed")
    add("CV8350", "PT8350", "local valve changes the local branch pressure", "high", 1, 8, "context_dependent", confirmed, "new_operator_confirmed")
    add("TE8353", "A管", "fluid passes TE8353 before the A-line temperature sensor", "high", 1, 12, "usually_positive", confirmed, "new_operator_confirmed")
    add("A管", "H_mix", "shared A-line thermal state enters the virtual downstream merge state", "high", 1, 30, "usually_positive", confirmed, "downstream_virtual_merge")
    add("EC-V2", "H_mix", "EC-V2 modulates one A-line downstream branch", "high", 1, 30, "context_dependent", confirmed, "downstream_virtual_merge")
    add("COOLDOWN", "H_mix", "COOLDOWN modulates one A-line downstream branch", "high", 1, 30, "context_dependent", confirmed, "downstream_virtual_merge")
    add("FC-V1", "H_mix", "FC-V1 modulates one A-line downstream branch", "high", 1, 30, "context_dependent", confirmed, "downstream_virtual_merge")
    add("H_mix", "Thv", "virtual merged thermal-hydraulic state drives mean module temperature", "high", 1, 30, "context_dependent", confirmed, "downstream_virtual_merge")

    # State persistence candidates. These are modeling priors, not operator claims.
    for signal, mechanism, lag_max in (
        ("TE8310", "thermal inertia", 30),
        ("PT8310", "pressure persistence", 15),
        ("TE8351", "thermal inertia", 30),
        ("PT8351", "pressure persistence", 15),
        ("FT8351", "flow persistence", 15),
        ("TE8352", "thermal inertia", 30),
        ("PT8352", "pressure persistence", 15),
        ("TE8353", "thermal inertia", 30),
        ("Thv", "module thermal inertia", 30),
    ):
        add(signal, signal, mechanism, "high", 1, lag_max, "usually_positive", old, "retained_self_lag")
    for signal, mechanism, lag_max in (
        ("TE8330", "thermal inertia", 30),
        ("PT8330", "pressure persistence", 15),
        ("TE8350", "thermal inertia", 30),
        ("PT8350", "pressure persistence", 15),
        ("A管", "thermal inertia", 30),
    ):
        add(signal, signal, mechanism, "medium", 1, lag_max, "usually_positive", persistence, "new_self_lag_proposed")

    dynamic_edge_id = "three_branch_kan_gate_8310_to_8351"
    for item in r:
        if (item["source"], item["destination"]) in {
            ("TE8310", "TE8351"),
            ("PT8310", "PT8351"),
        }:
            item["edge_function_id"] = dynamic_edge_id
            item["edge_function_mode"] = "KAN_dynamic_branch_gate"
    return r


def build_json(relations: list[dict[str, object]]) -> dict[str, object]:
    import pandas as pd

    catalog = pd.read_csv(CATALOG)
    raw_nodes = catalog.to_dict(orient="records")
    nodes = [
        {
            **node,
            "is_virtual": False,
            "directly_observed": True,
        }
        for node in raw_nodes
    ]
    nodes.append(
        {
            "node_index": len(raw_nodes),
            "signal": "H_mix",
            "category": "latent_thermal_hydraulic_state",
            "is_target": False,
            "is_future_control_in_0912": False,
            "is_virtual": True,
            "directly_observed": False,
        }
    )
    active = {str(r["source"]) for r in relations} | {str(r["destination"]) for r in relations}
    return {
        "schema_version": 2,
        "name": "HTC8300_raw54_static_physical_prior_draft",
        "status": "draft_for_operator_review",
        "sample_period_seconds": 10,
        "lag_semantics": "source(t-lag_steps) -> destination(t); lag 1 equals 10 seconds",
        "lag_warning": "lag_min and lag_max are candidate search windows, not delays estimated by PCMCI",
        "edge_index_rule": "Static edge_index keeps the physical TE8310->TE8351 and PT8310->PT8351 directions and adds A管/EC-V2/COOLDOWN/FC-V1->H_mix plus H_mix->Thv. KAN supplies the two dynamic edge functions and the virtual H_mix node update; information inputs into KAN are not interpreted as material-flow edges.",
        "scope_note": "All 54 Raw54 signals plus one unobserved virtual node H_mix are registered. A sparse static prior intentionally leaves undocumented raw signals without cross-variable physical edges; those signals remain available to the GRU and whole-data PCMCI screen.",
        "topology_confirmation": {
            "upstream_flow_split": {
                "branches": ["branch_1_CV8312", "branch_2_CV8311", "branch_3_CV8310"],
                "physical_merge": False,
                "coupling_mechanism": "flow_split_competition_only",
            },
            "downstream_of_A_line": {
                "parallel_valves": ["EC-V2", "COOLDOWN", "FC-V1"],
                "shared_upstream_state": "A管",
                "virtual_merge_node": "H_mix",
                "destination": "Thv",
                "physical_interpretation": "three A-line downstream valve paths jointly determine the aggregate state acting on Thv",
            },
            "confirmed_by": "operator_2026-09-12",
        },
        "dynamic_edge_functions": [
            {
                "edge_function_id": "three_branch_kan_gate_8310_to_8351",
                "model": "KAN",
                "semantics": "The three branch encodings modulate message strength; they do not create cross-branch material-flow edges.",
                "branch_inputs": {
                    "branch_1": ["CV8312", "CV8330", "TE8330", "PT8330"],
                    "branch_2": ["CV8311", "CV8350", "TE8350", "PT8350"],
                    "branch_3": ["CV8310", "CV8351", "TE8310", "PT8310"],
                },
                "governed_edges": [
                    {"source": "TE8310", "destination": "TE8351", "gate": "alpha_T(t)"},
                    {"source": "PT8310", "destination": "PT8351", "gate": "alpha_P(t)"},
                ],
                "lag_source": "PCMCI-selected stable training-only lags",
            }
        ],
        "virtual_state_functions": [
            {
                "function_id": "downstream_three_valve_virtual_merge",
                "node": "H_mix",
                "model": "KAN",
                "inputs": ["A管", "EC-V2", "COOLDOWN", "FC-V1"],
                "semantics": "H_mix is an unobserved aggregate state, not a measured temperature or flow. It represents the joint downstream effect of the shared A-line state and the three active valve openings.",
                "causal_direction_note": "A管 and the three valve-opening signals jointly update H_mix; no A管->valve-opening causal edge is created.",
                "lag_source": "PCMCI-selected stable training-only lags for observed inputs; H_mix itself is learned end-to-end",
                "outgoing_edge": {"source": "H_mix", "destination": "Thv"},
            }
        ],
        "nodes": nodes,
        "nodes_without_documented_static_relations": [
            row["signal"] for row in raw_nodes if row["signal"] not in active
        ],
        "removed_or_replaced_relations": [
            {
                "source": "CV8311",
                "destination": "FT8351",
                "reason": "Removed as a direct physical edge: the operator confirmed that the three branches do not merge; CV8311 participates only through flow-split competition and the KAN branch-state gate.",
            },
            {
                "source": "CV8312",
                "destination": "FT8351",
                "reason": "Removed as a direct physical edge: the operator confirmed that the three branches do not merge; CV8312 participates only through flow-split competition and the KAN branch-state gate.",
            },
            {
                "source": "TE8353",
                "destination": "Thv",
                "reason": "Replaced by the explicit mediated path TE8353 -> A管 -> H_mix -> Thv supplied on 2026-09-12.",
            },
            {
                "source": "A管",
                "destination": "Thv",
                "reason": "Replaced by A管 -> H_mix -> Thv so the three downstream valve branches can interact through an explicit virtual merge state.",
            },
            {
                "source": "EC-V2/COOLDOWN/FC-V1",
                "destination": "Thv",
                "reason": "Replaced by valve -> H_mix -> Thv; the valve-opening signals are active controls and are not modeled as effects of A管 temperature.",
            },
            {
                "source": "FT8351",
                "destination": "Thv",
                "reason": "Removed as a redundant shortcut because its influence is already mediated by the serial downstream path toward A管 and H_mix.",
            },
        ],
        "relations": relations,
    }


def draw_node(
    ax,
    x: float,
    y: float,
    label: str,
    kind: str = "sensor",
    width: float = 1.15,
    height: float = 0.56,
    fontsize: float = 5.8,
) -> None:
    colors = {
        "sensor": ("#E8F1F8", "#3775BA"),
        "valve": ("#FCE8D5", "#D97706"),
        "target": ("#DDF3DE", "#2E7D32"),
        "branch": ("#FFF6EC", "#D97706"),
        "kan": ("#EFE8F7", "#7A4FA3"),
        "latent": ("#E2F3F0", "#2A7F7A"),
        "note": ("#F4F4F4", "#767676"),
    }
    face, edge = colors[kind]
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#272727", zorder=4)


def arrow(
    ax,
    start,
    end,
    color,
    style="-",
    rad=0.0,
    label=None,
    label_offset=(0, 0),
    linewidth=1.05,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=linewidth,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=2,
    )
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2 + label_offset[0]
        y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(x, y, label, ha="center", va="center", fontsize=5.4, color=color, zorder=5)


def draw_figure() -> None:
    physical = "#5B7083"
    branch = "#D97706"
    information = "#7A4FA3"
    merge = "#2A7F7A"
    fig, ax = plt.subplots(figsize=(7.2047, 5.7087))  # 183 mm by 145 mm
    ax.set_xlim(0, 15.6)
    ax.set_ylim(-0.9, 10.6)
    ax.axis("off")

    ax.text(0.25, 10.27, "HTC8300 静态物理关系 v2（分流门控 + 虚拟汇合确认稿）", fontsize=9, fontweight="bold", ha="left")
    ax.text(
        0.25,
        9.84,
        "上游三支路不发生物料汇合；A管后的 EC-V2、COOLDOWN、FC-V1 共同形成潜在汇合状态 H_mix，再作用于 Thv",
        fontsize=5.7,
        color="#606060",
        ha="left",
    )

    # Three independent branch-state encoders.
    draw_node(ax, 1.8, 8.3, "支路 1（独立）\nCV8312 / CV8330\nTE8330 / PT8330", "branch", 2.85, 1.05, 5.5)
    draw_node(ax, 1.8, 6.8, "支路 2（独立）\nCV8311 / CV8350\nTE8350 / PT8350", "branch", 2.85, 1.05, 5.5)
    draw_node(ax, 1.8, 5.3, "支路 3（独立）\nCV8310 / CV8351\nTE8310 / PT8310", "branch", 2.85, 1.05, 5.5)
    draw_node(ax, 5.9, 6.8, "KAN 分流边函数\n三支路状态编码\n输出 αT(t)、αP(t)", "kan", 2.4, 2.0, 6.0)
    arrow(ax, (3.23, 8.3), (4.7, 7.4), information, style="--", rad=-0.08)
    arrow(ax, (3.23, 6.8), (4.7, 6.8), information, style="--")
    arrow(ax, (3.23, 5.3), (4.7, 6.2), information, style="--", rad=0.08)
    ax.text(3.95, 7.03, "状态信息", fontsize=5.3, color=information, ha="center")
    draw_node(ax, 10.55, 7.7, "已确认：上游三支路不汇合\n仅通过三个主分流阀形成耦合", "note", 3.45, 1.05, 5.8)
    ax.text(8.82, 6.78, "无支路1/2 → 主支路的物料边", fontsize=5.4, color="#8B3A3A", ha="left")

    # Main physical propagation; CV8310 is not placed downstream of TE/PT8310.
    ax.text(0.3, 4.15, "主物理传播路径", fontsize=6.4, fontweight="bold", color="#3775BA")
    draw_node(ax, 0.8, 3.2, "CV8300\nCV8313", "valve", 1.05, 0.72)
    draw_node(ax, 2.3, 3.2, "TE8310\nPT8310", "sensor", 1.12, 0.72)
    draw_node(ax, 5.55, 3.2, "TE8351\nPT8351", "sensor", 1.17, 0.72)
    draw_node(ax, 5.55, 2.15, "FT8351", "sensor", 0.92, 0.58)
    draw_node(ax, 7.4, 3.2, "TE8352\nPT8352", "sensor", 1.08, 0.72)
    draw_node(ax, 8.92, 3.2, "TE8353", "sensor", 0.88, 0.62)
    draw_node(ax, 10.05, 3.2, "A管", "sensor", 0.68, 0.62)
    draw_node(ax, 12.15, 3.2, "H_mix\n虚拟汇合状态\nKAN 聚合", "latent", 1.58, 0.92, 5.3)
    draw_node(ax, 14.6, 3.2, "Thv\n模组平均温度", "target", 1.5, 0.78)
    arrow(ax, (1.33, 3.2), (1.74, 3.2), physical)
    arrow(ax, (2.86, 3.2), (4.96, 3.2), information, linewidth=2.4)
    ax.text(3.45, 2.43, "KAN边函数 αT(t)、αP(t)", fontsize=5.5, color=information, ha="center")
    arrow(ax, (5.42, 5.82), (3.95, 3.3), information, style="--", rad=0.12)
    arrow(ax, (6.14, 3.2), (6.86, 3.2), physical)
    arrow(ax, (7.94, 3.2), (8.48, 3.2), physical)
    arrow(ax, (9.36, 3.2), (9.71, 3.2), physical)
    arrow(ax, (10.39, 3.2), (11.36, 3.2), merge, linewidth=1.7)
    arrow(ax, (12.94, 3.2), (13.85, 3.2), merge, linewidth=1.7)
    arrow(ax, (6.01, 2.15), (6.91, 2.85), physical, rad=-0.1)

    # Module-side three-valve split represented by one virtual merge state.
    ax.text(12.15, 4.15, "A管后三路分流 → 潜在汇合", fontsize=6.1, fontweight="bold", color=merge, ha="center")
    draw_node(ax, 10.55, 1.15, "EC-V2", "valve", 1.0, 0.58)
    draw_node(ax, 12.15, 1.15, "COOLDOWN", "valve", 1.60, 0.58)
    draw_node(ax, 13.75, 1.15, "FC-V1", "valve", 1.0, 0.58)
    arrow(ax, (10.72, 1.44), (11.66, 2.74), merge, rad=-0.08, linewidth=1.35)
    arrow(ax, (12.15, 1.44), (12.15, 2.74), merge, linewidth=1.35)
    arrow(ax, (13.58, 1.44), (12.64, 2.74), merge, rad=0.08, linewidth=1.35)
    ax.text(12.15, 0.58, "H_mix 为不可观测潜在状态；不建立 A管温度 → 阀门开度的因果边", fontsize=5.1, color="#606060", ha="center")

    legend = [
        Line2D([0], [0], color=physical, lw=1.4, label="物理传播边"),
        Line2D([0], [0], color=information, lw=1.4, ls="--", label="支路状态编码（非物料边）"),
        Line2D([0], [0], color=information, lw=2.4, label="KAN 动态边函数"),
        Line2D([0], [0], color=merge, lw=1.7, label="下游虚拟汇合路径"),
    ]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0.01, 0.002), ncol=4, fontsize=5.35, frameon=False)
    ax.text(15.35, -0.48, "时延由 PCMCI 在训练集上筛选；自滞后边未绘出", ha="right", va="bottom", fontsize=5.1, color="#606060")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base = FIGURE_DIR / "static_physical_graph_v2_review"
    if STANDALONE_FIGURE_ONLY:
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
    else:
        if require_matplotlib_panel_alignment is None:
            raise RuntimeError("nature-figure alignment helper is unavailable")
        require_matplotlib_panel_alignment(
            fig,
            axes=[ax],
            panel_ids=["a"],
            json_out=str(base) + ".alignment.json",
            overlay_svg=str(base) + ".alignment.svg",
            strict=True,
        )
        fig.savefig(str(base) + ".svg", bbox_inches="tight")
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_review(graph: dict[str, object]) -> None:
    import pandas as pd

    relations = pd.DataFrame(graph["relations"])
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    json_path = GRAPH_DIR / "static_physical_graph_v2_draft.json"
    csv_path = GRAPH_DIR / "static_physical_graph_v2_edge_review.csv"
    json_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    relations.to_csv(csv_path, index=False, encoding="utf-8-sig")

    cross = relations[relations["source"] != relations["destination"]]
    self_lag = relations[relations["source"] == relations["destination"]]
    markdown = f"""# HTC8300 静态物理关系 v2 审核说明

## 已确认的核心拓扑

- 上游由 `CV8312、CV8311、CV8310` 控制的三条支路彼此独立，**后续不会发生物料汇合**。
- 三条支路之间的耦合来自 `CV8312`、`CV8311`、`CV8310` 的分流关系，而不是支路传感器之间的直接物理连接。
- `TE8310/PT8310` 不指向控制阀 `CV8310`；`CV8310` 是主动控制输入。
- `TE8310 -> TE8351` 与 `PT8310 -> PT8351` 是物理传播方向，但边函数由三支路状态编码后的 KAN 动态构造。
- A管后的 `EC-V2、COOLDOWN、FC-V1` 是另一组下游并联分流阀；它们最终共同作用于 Thv，因此增加不可观测虚拟节点 `H_mix` 表示等效汇合状态。
- Raw54 的 54 个观测信号全部登记，并额外增加 1 个虚拟节点。当前共有 **{len(relations)} 条关系模板**：{len(cross)} 条跨变量物理/虚拟关系、{len(self_lag)} 条状态自滞后关系。

## 三支路 KAN 边函数

三个支路编码输入暂定为：

1. 支路1：`CV8312、CV8330、TE8330、PT8330`。
2. 支路2：`CV8311、CV8350、TE8350、PT8350`。
3. 支路3：`CV8310、CV8351、TE8310、PT8310`。

KAN 输出两个动态门控系数：

- `alpha_T(t)` 调节 `TE8310(t-lag_T) -> TE8351(t)`。
- `alpha_P(t)` 调节 `PT8310(t-lag_P) -> PT8351(t)`。

这些“支路状态 -> KAN”的虚线只代表模型信息输入，不写入物理 `edge_index`，也不表示物料汇合。静态 `edge_index` 仍保存两条上游到下游的物理方向；KAN 负责计算其随工况变化的边权/消息函数。

## A管后的虚拟汇合节点

下游结构表示为：

`A管、EC-V2、COOLDOWN、FC-V1 -> H_mix -> Thv`

节点更新暂定为：

`H_mix(t) = KAN[A管(t-lag_A), EC-V2(t-lag_EC), COOLDOWN(t-lag_CD), FC-V1(t-lag_FC)]`

- `H_mix` 是模型学习的潜在热力/流动状态，不是实际测量的温度、压力或流量。
- `A管` 与三个阀门开度共同输入 `H_mix`，但不建立 `A管温度 -> 阀门开度` 的因果边。
- `A管 -> Thv` 与三个阀门分别直接指向 `Thv` 的旧捷径边被替换，避免模型绕过三阀门的联合效应。
- 这组 KAN 可学习阀门之间的交互，以及相同阀门开度在不同 A管温度下产生的不同作用。

## 局部关系仍然保留

- `CV8312/CV8330 -> TE8330、PT8330`。
- `CV8311/CV8350 -> TE8350、PT8350`。
- `TE8353 -> A管 -> H_mix -> Thv`。
- `EC-V2、COOLDOWN、FC-V1 -> H_mix`。

旧版推测性的 `CV8311 -> FT8351` 和 `CV8312 -> FT8351` 已从直接物理边中删除，因为支路已确认不汇合。它们对主路的影响只通过分流状态和 KAN 门控表达。

## 时延与预测时的读取规则

- `lag_min/lag_max` 仍只是 PCMCI 的候选搜索区间，不是最终时延。
- PCMCI 只在训练集上分别确定 `lag_T`、`lag_P`、各上游支路编码变量，以及 `A管/EC-V2/COOLDOWN/FC-V1` 的稳定滞后。
- 预测时，KAN 只能读取预测起点以前的支路温度/压力；未来阀门开度只有在控制计划已知时才能进入 future-control 分支。
- `H_mix` 没有直接观测值，因此不由 PCMCI 直接筛选，而是在预测损失监督下端到端学习。
- PCMCI 不负责判断支路是否汇合；上游不汇合、A管后三阀门共同汇合到 `H_mix` 均由人工物理知识固定。

## 后续仍需实验确定

1. KAN 对 `alpha_T` 和 `alpha_P` 使用一个共享网络还是两个独立输出头。
2. 支路局部阀 `CV8330/CV8350/CV8351` 是否全部进入编码，还是由 PCMCI 稳定性筛选后保留。
3. 门控系数按当前时刻生成一个值，还是为未来 15 个预测步分别生成 15 组值。
4. `FC-V1` 是否存在未来 150 s 的可用计划开度。
5. `H_mix` 使用单时刻 KAN 聚合，还是加入一层 GRU 保存潜在状态的短期记忆。
"""
    (DOC_DIR / "static_physical_graph_v2_review.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    if STANDALONE_FIGURE_ONLY:
        draw_figure()
        print(json.dumps({
            "mode": "standalone_figure_only",
            "output": str(FIGURE_DIR / "static_physical_graph_v2_review.png"),
        }, ensure_ascii=False, indent=2))
        return

    relations = build_relations()
    graph = build_json(relations)
    write_review(graph)
    draw_figure()
    print(json.dumps({
        "relations": len(relations),
        "cross_variable": sum(r["source"] != r["destination"] for r in relations),
        "self_lag": sum(r["source"] == r["destination"] for r in relations),
        "new_operator_confirmed": sum(r["relation_group"] == "new_operator_confirmed" for r in relations),
        "nodes": len(graph["nodes"]),
        "unconnected_nodes": len(graph["nodes_without_documented_static_relations"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
