#!/usr/bin/env python3
"""Reproduce the completed architecture comparison using the embedded real seed results.

Dependencies: numpy, pandas, scipy, matplotlib, pillow.
Run: python model_comparison_0906.py
Optional: --output-dir PATH --qa-dir PATH --qa-tools PATH
The embedded table contains all 85 completed runs; no seed is removed.
Errors concern restored Thv(t+15) at all 10,659 valid rolling test origins.
This is not a single-initialization open-loop trajectory evaluation.
"""
from __future__ import annotations
import argparse
from io import StringIO
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Observed results, never illustrative or synthesized data.
SEED_RESULTS_CSV = """model,seed,test_full_rmse_k,validation_rmse_k,test_full_mae_k,trainable_parameters
gru_baseline,42,1.168597160074331,0.1974083901620981,0.2855444691864223,26657
gru_baseline,52,1.207855682276126,0.1937841244720249,0.3018865429933627,26657
gru_baseline,62,1.139963722268742,0.2020273859926166,0.3147729798720604,26657
gru_baseline,72,1.1589226087435374,0.2071508915639517,0.3154295423899961,26657
gru_baseline,82,1.1789252246643402,0.1886272096675726,0.3128644420745565,26657
gru_baseline,92,1.182240601471859,0.2063045061813441,0.3049347665996496,26657
gru_baseline,102,1.1291828105316777,0.2037892107025292,0.3664967393050215,26657
gru_baseline,112,1.2348566251812148,0.2063869498322418,0.3551069345261746,26657
gru_baseline,122,1.1568188263207708,0.2050748749176265,0.344668790014584,26657
gru_baseline,132,1.20551674056881,0.2109369451762478,0.3237289001268398,26657
lstm_baseline,42,1.2322160888893243,0.2506498882094243,0.3455579242055406,32161
lstm_baseline,52,1.238651864929294,0.2231477585041107,0.3641996522532718,32161
lstm_baseline,62,1.1881735105344111,0.2357561474377575,0.3489319269290379,32161
lstm_baseline,72,1.201890122148329,0.2155100097676451,0.3009800287133672,32161
lstm_baseline,82,1.1127717187380963,0.2298145892107038,0.2932138779869193,32161
lstm_baseline,92,1.131731338590252,0.2400254611414449,0.3146053725152732,32161
lstm_baseline,102,1.1839547995167905,0.2240616992305679,0.3108930988822229,32161
lstm_baseline,112,1.198629899643556,0.2205409002530687,0.406151878601007,32161
lstm_baseline,122,1.2696025555841417,0.2687217776060028,0.3496251802906511,32161
lstm_baseline,132,1.1708240160790104,0.2431060181434699,0.3353808831973144,32161
parallel_gru_kan_gnn,42,1.179282423572391,0.1896407031727561,0.2797375992755949,87193
parallel_gru_kan_gnn,52,1.157174656449404,0.1976803256285899,0.2986941338133443,87193
parallel_gru_kan_gnn,62,1.0797136768945834,0.2044721426556494,0.2758947503045291,87193
parallel_gru_kan_gnn,72,1.1841979158941194,0.2029133332841435,0.3097007337741834,87193
parallel_gru_kan_gnn,82,1.1324336768418588,0.2055229239219635,0.2825357867937282,87193
parallel_gru_kan_gnn,92,1.1592904399940642,0.195329978759029,0.2998329346512315,87193
parallel_gru_kan_gnn,102,1.141548581169903,0.1960876434895099,0.284164027007221,87193
parallel_gru_kan_gnn,112,1.2282997442664922,0.1967888930360449,0.3101673822441612,87193
parallel_gru_kan_gnn,122,1.1470247064520618,0.203980382428781,0.2951631132806629,87193
parallel_gru_kan_gnn,132,1.0896452127734273,0.2104068099779455,0.3011584707658255,87193
parallel_gru_mlp_gnn,42,1.207647201717912,0.1917129940888608,0.3018188540483876,87059
parallel_gru_mlp_gnn,52,1.1724382255165875,0.1985861784604637,0.3010843392981622,87059
parallel_gru_mlp_gnn,62,1.1269678884692602,0.2018371638511627,0.2966587488917838,87059
parallel_gru_mlp_gnn,72,1.1612342189186495,0.1891027921106756,0.3091558320082035,87059
parallel_gru_mlp_gnn,82,1.130607382688764,0.2011917215124009,0.2789126743421466,87059
parallel_gru_mlp_gnn,92,1.0323140554411614,0.1984845498688975,0.2794899658156309,87059
parallel_gru_mlp_gnn,102,1.1042232652351625,0.198635324475771,0.3160657455402932,87059
parallel_gru_mlp_gnn,112,1.192249459041971,0.1904990746118443,0.3168656341186618,87059
parallel_gru_mlp_gnn,122,1.1408744254537388,0.189848205780977,0.3073866639967967,87059
parallel_gru_mlp_gnn,132,1.150679079514373,0.1979717109132753,0.3106227603100925,87059
parallel_lstm_kan_gnn,42,1.1472988721075306,0.242027985535027,0.2902268625125644,92697
parallel_lstm_kan_gnn,52,1.135308367186692,0.2298073566030263,0.2992836253196699,92697
parallel_lstm_kan_gnn,62,1.124403991869212,0.2150087412368042,0.2870027135550841,92697
parallel_lstm_kan_gnn,72,1.126706546912193,0.2146251855339454,0.3179622836818053,92697
parallel_lstm_kan_gnn,82,1.071440850026825,0.2205704743804284,0.2942684335524077,92697
parallel_lstm_kan_gnn,92,1.1472824337928185,0.1988301010787515,0.2952962956697638,92697
parallel_lstm_kan_gnn,102,1.1110460422490611,0.2133305947973975,0.2733051008597985,92697
parallel_lstm_kan_gnn,112,1.21241988074228,0.2047739235658659,0.3190104972609138,92697
parallel_lstm_kan_gnn,122,1.1926390449174882,0.2331726602050001,0.3228781967383706,92697
parallel_lstm_kan_gnn,132,1.2509327957027534,0.2126813216870598,0.3195585840557431,92697
parallel_lstm_mlp_gnn,42,1.128692704133725,0.2132171138890097,0.3161674869152689,92563
parallel_lstm_mlp_gnn,52,1.2343386311500288,0.2200308238146404,0.3154934704309661,92563
parallel_lstm_mlp_gnn,62,1.1055821373977923,0.2138561666387056,0.294434582366422,92563
parallel_lstm_mlp_gnn,72,1.0560299010919112,0.2166284389015509,0.2961189925223391,92563
parallel_lstm_mlp_gnn,82,1.1971835319318478,0.2222730494666052,0.3129988978639679,92563
parallel_lstm_mlp_gnn,92,1.1858146462889658,0.2272814802677278,0.3136961183535087,92563
parallel_lstm_mlp_gnn,102,1.0704979089773523,0.2416682608350946,0.2919087810608249,92563
parallel_lstm_mlp_gnn,112,1.1945659370206387,0.2069894229181711,0.3277113769524098,92563
parallel_lstm_mlp_gnn,122,1.1874437917723997,0.2122451679553506,0.3045327754025184,92563
parallel_lstm_mlp_gnn,132,1.090920450921105,0.2582248838012658,0.3103374836951637,92563
serial_kan_gnn_gru,42,2.1029767029160475,0.2023208917233294,0.7177372310742945,91545
serial_kan_gnn_gru,52,2.196375789521656,0.1967310573958204,0.7646747621732948,91545
serial_kan_gnn_gru,62,2.0114153192050144,0.1857466713550833,0.6455541962160016,91545
serial_kan_gnn_gru,72,2.0782529668910428,0.1888264604946315,0.7236228132038813,91545
serial_kan_gnn_gru,82,2.0185773058762027,0.1880454550049755,0.8317253086277612,91545
serial_kan_gnn_gru,92,2.6542596688941065,0.1863251103279903,1.0994162859196914,91545
serial_kan_gnn_gru,102,2.0819933244828706,0.1838906397891717,0.7521026302052197,91545
serial_kan_gnn_gru,112,2.202807903738458,0.186962257917197,0.726494643879921,91545
serial_kan_gnn_gru,122,2.1945649481538445,0.1833202506665471,0.7164744721496182,91545
serial_kan_gnn_gru,132,1.9948102888358008,0.1847082150863864,0.6730367617469145,91545
serial_mlp_gnn_gru,42,2.1188933735242728,0.1922092166352963,0.7716847830526239,91411
serial_mlp_gnn_gru,52,2.094584109526689,0.191346599560151,0.8162816176645764,91411
serial_mlp_gnn_gru,62,2.069802822381905,0.1869565410878649,0.7082840005597424,91411
serial_mlp_gnn_gru,72,2.090076553335152,0.1863781102743878,0.7747241624787281,91411
serial_mlp_gnn_gru,82,2.298126183144184,0.185886462400355,0.8095116790525511,91411
serial_mlp_gnn_gru,92,2.180176709216312,0.1913880476834942,0.8662990759379591,91411
serial_mlp_gnn_gru,102,2.108991436325337,0.1937459895267308,0.8466830492038021,91411
serial_mlp_gnn_gru,112,4.589997274040132,0.190599932208007,2.374263660096785,91411
serial_mlp_gnn_gru,122,2.171759655014188,0.2012548519056686,0.7129095552495379,91411
serial_mlp_gnn_gru,132,2.084583586899939,0.1948045122281185,0.6782625713213756,91411
tcn_baseline,42,1.33799018630774,0.2126766136512654,0.3490074987889128,86369
tcn_baseline,62,1.3296046137746005,0.2173262288660654,0.3356318749823775,86369
tcn_baseline,82,1.326426430436791,0.2081723325139618,0.3236848074639077,86369
tcn_baseline,102,1.433742421734341,0.1987703220145311,0.3930297469734005,86369
tcn_baseline,122,1.2472427572198237,0.1916491290718836,0.3114500591118407,86369
"""
PERSISTENCE_RMSE_K = 1.9780479142129992
LABELS = {
    "gru_baseline": "GRU",
    "lstm_baseline": "LSTM",
    "tcn_baseline": "TCN",
    "parallel_gru_mlp_gnn": "Parallel GRU–MLP–GNN",
    "parallel_gru_kan_gnn": "Parallel GRU–KAN–GNN",
    "parallel_lstm_mlp_gnn": "Parallel LSTM–MLP–GNN",
    "parallel_lstm_kan_gnn": "Parallel LSTM–KAN–GNN",
    "serial_mlp_gnn_gru": "Serial MLP–GNN–GRU",
    "serial_kan_gnn_gru": "Serial KAN–GNN–GRU",
}
COLORS = {"baseline": "#A4ABB3", "parallel": "#82B7D8", "serial": "#DDB084"}
DARK = "#253344"

def family_color(model):
    return COLORS["parallel" if model.startswith("parallel") else
                  "serial" if model.startswith("serial") else "baseline"]

def configure():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
    })

def save_figure(fig, name, args):
    fig.canvas.draw()
    if args.qa_tools:
        sys.path.insert(0, str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        args.qa_dir.mkdir(parents=True, exist_ok=True)
        require_matplotlib_panel_alignment(
            fig, json_out=str(args.qa_dir / (name + ".alignment.json")),
            overlay_svg=str(args.qa_dir / (name + ".alignment.svg")),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    # Fixed dimensions: no tight cropping that changes the publication width.
    fig.savefig(args.output_dir / (name + ".pdf"))
    fig.savefig(args.output_dir / (name + ".svg"))
    fig.savefig(args.output_dir / (name + ".png"), dpi=600)
    fig.savefig(args.output_dir / (name + ".tiff"), dpi=1000,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

def base_axes(height_mm, right=0.82):
    fig, ax = plt.subplots(figsize=(183/25.4, height_mm/25.4))
    fig.subplots_adjust(left=0.335, right=right, top=0.84, bottom=0.245)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E4E8EC", linewidth=0.45)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3, width=0.6)
    return fig, ax

def main_figure(data, summary, args):
    models = summary.index.tolist()
    fig, ax = base_axes(122)
    y = np.arange(len(models))
    for i, model in enumerate(models):
        values = data.loc[data.model == model, "test_full_rmse_k"].to_numpy()
        mean, sd = summary.loc[model, ["mean","sd"]]
        ax.barh(i, mean, height=0.55, color=family_color(model),
                edgecolor="none", zorder=2)
        # All seed points have exactly the row's y coordinate: no jitter.
        ax.scatter(values, np.full(len(values),i), s=10, facecolors="white",
                   edgecolors=DARK, linewidths=0.45, zorder=4)
        ax.errorbar(mean, i, xerr=sd, fmt="D", markersize=3.2,
                    color=DARK, ecolor=DARK, capsize=2.3,
                    elinewidth=0.8, markeredgewidth=0.6, zorder=5)
        ax.text(1.035, i, f"{mean:.3f} ± {sd:.3f}",
                transform=ax.get_yaxis_transform(), va="center", fontsize=7.6)
    ax.axvline(PERSISTENCE_RMSE_K, color="#667681",
               linestyle=(0,(4,3)), linewidth=0.85, zorder=3)
    ax.set_yticks(y, [f"{LABELS[m]}  (n={int(summary.loc[m,'n'])})" for m in models])
    ax.set_ylim(len(models)-0.45,-0.65)
    ax.set_xlim(0,4.9)
    ax.set_xticks(np.arange(0,5,1))
    ax.set_xlabel("Full-test RMSE (K)",labelpad=8)
    fig.text(0.335,0.925,"Complete test sequence",fontsize=10,fontweight="bold")
    fig.text(0.335,0.878,"Bars: mean ± SD; points: individual seeds",fontsize=7.5)
    fig.text(0.837,0.878,"RMSE (K)",fontsize=7.5)
    handles = [
        Patch(facecolor=COLORS["baseline"],label="Temporal baseline"),
        Patch(facecolor=COLORS["parallel"],label="Parallel"),
        Patch(facecolor=COLORS["serial"],label="Serial"),
    ]
    fig.legend(handles=handles, loc="lower center",bbox_to_anchor=(0.61,0.067),
               ncol=3,frameon=False,fontsize=7.5,handlelength=1.15,
               columnspacing=1.3)
    fig.legend(handles=[Line2D([0],[0],color="#667681",ls=(0,(4,3)),lw=.85,
                   label=f"Persistence: {PERSISTENCE_RMSE_K:.3f} K")],
               loc="lower center",bbox_to_anchor=(0.61,0.021),
               frameon=False,fontsize=7.5)
    save_figure(fig,"model_comparison_0906",args)

def paired_figure(data, summary, args):
    models = [m for m in summary.index if m!="gru_baseline"]
    baseline = data.loc[data.model=="gru_baseline"].set_index("seed").test_full_rmse_k
    fig, ax = base_axes(118, right=0.755)
    effects=[]
    for i, model in enumerate(models):
        # TCN uses only the same five seeds in the GRU comparison.
        x = data.loc[data.model==model].set_index("seed").test_full_rmse_k
        assert x.index.is_unique and x.index.isin(baseline.index).all()
        d = (x - baseline.loc[x.index]).to_numpy()
        assert len(d) == len(x) and np.isfinite(d).all()
        mean = float(d.mean())
        half = float(stats.t.ppf(.975,len(d)-1)*stats.sem(d))
        ax.errorbar(mean,i,xerr=half,fmt="D",markersize=4,
                    color=DARK,markerfacecolor=family_color(model),
                    capsize=3,elinewidth=1,markeredgewidth=.65,zorder=4)
        ax.text(1.04,i,f"{mean:+.3f} [{mean-half:+.3f}, {mean+half:+.3f}]",
                transform=ax.get_yaxis_transform(),va="center",fontsize=7.1)
        effects.append({"model":model,"n":len(d),"mean_difference_k":mean,
                        "ci95_low_k":mean-half,"ci95_high_k":mean+half})
    ax.axvline(0,color="#667681",ls=(0,(4,3)),lw=.8,zorder=2)
    ax.set_yticks(np.arange(len(models)),
                 [f"{LABELS[m]}  (n={int(summary.loc[m,'n'])})" for m in models])
    ax.set_ylim(len(models)-.45,-.65)
    ax.set_xlim(-.13,1.85)
    ax.set_xticks([0,.5,1,1.5])
    ax.set_xlabel("RMSE difference from GRU (K)",labelpad=8)
    fig.text(.335,.925,"Paired comparison with GRU",fontsize=10,fontweight="bold")
    fig.text(.335,.878,"Mean difference and 95% CI",fontsize=7.5)
    fig.text(.772,.878,"Difference [95% CI] (K)",fontsize=7.1)
    fig.text(.335,.105,"Negative differences favor the compared model.",fontsize=7.5)
    fig.text(.335,.056,"Seeds are paired; intervals reflect initialization variability.",fontsize=7.3)
    save_figure(fig,"model_comparison_0906_paired",args)
    return effects

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--qa-dir",type=Path)
    parser.add_argument("--qa-tools",type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.qa_tools and args.qa_dir is None:
        parser.error("--qa-tools requires --qa-dir")
    data=pd.read_csv(StringIO(SEED_RESULTS_CSV))
    assert len(data)==85 and not data.duplicated(["model","seed"]).any()
    assert set(data.model)==set(LABELS)
    for model, sub in data.groupby("model"):
        expected={42,62,82,102,122} if model=="tcn_baseline" else set(range(42,133,10))
        assert set(sub.seed)==expected
    # Average run-level RMSE, not RMSE from a seed-averaged ensemble prediction.
    summary=data.groupby("model").test_full_rmse_k.agg(
        mean="mean",sd="std",n="count").sort_values("mean")
    configure()
    main_figure(data,summary,args)
    effects=paired_figure(data,summary,args)
    if args.qa_dir:
        args.qa_dir.mkdir(parents=True,exist_ok=True)
        summary.to_csv(args.qa_dir/"plotted_model_summary.csv")
        data.to_csv(args.qa_dir/"plotted_seed_source.csv",index=False)
        (args.qa_dir/"plotted_paired_effects.json").write_text(
            json.dumps(effects,indent=2),encoding="utf-8")
    print(summary.to_string())
    print("FIGURES_COMPLETE: 2 figures, each PDF/SVG/PNG/TIFF; no seed exclusions.")

if __name__=="__main__":
    main()
