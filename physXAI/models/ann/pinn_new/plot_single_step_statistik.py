import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


parser = argparse.ArgumentParser(
    description="Erstellt die Single-Step-Diagramme für W28T und W14T."
)

parser.add_argument(
    "--base-dir",
    type=Path,
    default=Path(
        r"D:\phe-dwe\Git\testhall_offices_experiment"
        r"\0_agentlib_configs\results\sweep_single_step"
    ),
    help="Basisordner, der die Unterordner W28T und W14T enthält",
)

parser.add_argument(
    "--w28-dir",
    type=Path,
    default=None,
    help="Optionaler abweichender Ordner für W28T",
)

parser.add_argument(
    "--w14-dir",
    type=Path,
    default=None,
    help="Optionaler abweichender Ordner für W14T",
)

parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help=(
        "Pfad der Ausgabedatei. Standard: "
        "<base-dir>/single_step_ergebnisse_graphen.png"
    ),
)

args = parser.parse_args()


BASE_DIR = args.base_dir.resolve()

DATASET_DIRS = {
    "w28t": (
        args.w28_dir.resolve()
        if args.w28_dir is not None
        else BASE_DIR / "W28T"
    ),
    "w14t": (
        args.w14_dir.resolve()
        if args.w14_dir is not None
        else BASE_DIR / "W14T"
    ),
}

OUTPUT = (
    args.output.resolve()
    if args.output is not None
    else BASE_DIR / "single_step_ergebnisse_graphen.png"
)



LAMBDAS = [0.0, 0.2, 0.5, 0.7, 1.0, 3.0]

MODELS = [
    "1R1C",
    "2R2C",
    "Gokhale",
]

COLORS = {
    "ANN": "#4D4D4D",
    "1R1C": "#0072B2",
    "2R2C": "#009E73",
    "Gokhale": "#D55E00",
}

MARKERS = {
    "1R1C": "o",
    "2R2C": "s",
    "Gokhale": "^",
}


# ============================================================
# Hilfsfunktionen
# ============================================================

def comma_tick(value, _position=None):
    return f"{value:.2f}".replace(".", ",")


def find_csv(folder, filenames):
    for filename in filenames:
        path = folder / filename

        if path.exists():
            return path

    expected = "\n".join(
        f"  {folder / filename}"
        for filename in filenames
    )

    raise FileNotFoundError(
        "Keine passende CSV-Datei gefunden. "
        f"Geprüft wurden:\n{expected}"
    )


def load_data(tag):
    folder = DATASET_DIRS[tag]

    summary_path = find_csv(
        folder,
        [
            f"single_step_summary_{tag}.csv",
            "single_step_summary.csv",
        ],
    )

    results_path = find_csv(
        folder,
        [
            f"single_step_results_{tag}.csv",
            "single_step_results.csv",
        ],
    )

    print(f"{tag.upper()} Summary: {summary_path}")
    print(f"{tag.upper()} Results: {results_path}")

    summary = pd.read_csv(summary_path)
    results = pd.read_csv(results_path)

    summary = summary[
        summary["model"] != "Gokhale_wall_dynamics"
    ].copy()

    results = results[
        results["model"] != "Gokhale_wall_dynamics"
    ].copy()

    return summary, results


summaries = {}
results = {}

for dataset in ("w28t", "w14t"):
    summaries[dataset], results[dataset] = load_data(dataset)



sns.set_theme(
    style="whitegrid",
    context="notebook",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "semibold",
    "axes.labelweight": "normal",
    "legend.frameon": False,
})


fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(16, 12),
    gridspec_kw={
        "height_ratios": [1.0, 1.25],
    },
)


for ax, dataset, title in zip(
    axes[0],
    ("w28t", "w14t"),
    (
        "(a) W28T – Mittelwert ± Standardabweichung",
        "(b) W14T – Mittelwert ± Standardabweichung",
    ),
):
    summary = summaries[dataset]

    # ANN-Ergebnisse auslesen
    ann = summary[
        summary["model"] == "ANN"
    ].iloc[0]

    ann_mean = ann["test_rmse_mean"]
    ann_std = ann["test_rmse_std"]

    ax.axhspan(
        ann_mean - ann_std,
        ann_mean + ann_std,
        color=COLORS["ANN"],
        alpha=0.09,
        linewidth=0,
    )

    ax.axhline(
        y=ann_mean,
        color=COLORS["ANN"],
        linestyle=(0, (6, 4)),
        linewidth=2.1,
        label="ANN",
    )

    for model in MODELS:
        model_data = (
            summary[
                summary["model"] == model
            ]
            .set_index("lambda")
            .loc[LAMBDAS]
        )

        ax.errorbar(
            x=np.arange(len(LAMBDAS)),
            y=model_data["test_rmse_mean"],
            yerr=model_data["test_rmse_std"],
            color=COLORS[model],
            marker=MARKERS[model],
            markersize=6.5,
            linewidth=2.2,
            elinewidth=1.35,
            capsize=4,
            capthick=1.35,
            label=model,
        )

    ax.set_title(
        title,
        pad=12,
    )

    ax.set_xlabel("Gewichtung λ")
    ax.set_ylabel("Test-RMSE [K]")

    ax.set_xticks(
        np.arange(len(LAMBDAS))
    )

    ax.set_xticklabels(
        ["0", "0,2", "0,5", "0,7", "1", "3"]
    )

    ax.set_ylim(0.04, 0.40)

    ax.yaxis.set_major_formatter(
        FuncFormatter(comma_tick)
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)


handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.910),
    ncol=4,
    handlelength=2.5,
    columnspacing=2.0,
)



selected_configurations = [
    ("ANN", None, "ANN"),

    ("1R1C", 0.0, "1R1C λ=0"),
    ("1R1C", 0.5, "1R1C λ=0,5"),
    ("1R1C", 1.0, "1R1C λ=1"),

    ("2R2C", 0.0, "2R2C λ=0"),
    ("2R2C", 0.5, "2R2C λ=0,5"),
    ("2R2C", 1.0, "2R2C λ=1"),

    ("Gokhale", 0.0, "Gokhale λ=0"),
    ("Gokhale", 0.5, "Gokhale λ=0,5"),
    ("Gokhale", 1.0, "Gokhale λ=1"),
]


configuration_order = [
    label
    for _, _, label in selected_configurations
]


palette = {
    label: COLORS[model]
    for model, _, label in selected_configurations
}


for ax, dataset, title in zip(
    axes[1],
    ("w28t", "w14t"),
    (
        "(c) W28T – Verteilung der einzelnen Runs",
        "(d) W14T – Verteilung der einzelnen Runs",
    ),
):
    raw_results = results[dataset]

    selected_frames = []

    for model, lambda_value, label in selected_configurations:

        if lambda_value is None:
            part = raw_results[
                raw_results["model"] == model
            ].copy()

        else:
            part = raw_results[
                (
                    raw_results["model"] == model
                )
                & (
                    raw_results["lambda"] == lambda_value
                )
            ].copy()

        part["Konfiguration"] = label

        selected_frames.append(
            part[
                [
                    "Konfiguration",
                    "test_rmse",
                ]
            ]
        )

    boxplot_data = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    sns.boxplot(
        data=boxplot_data,
        x="test_rmse",
        y="Konfiguration",
        order=configuration_order,
        hue="Konfiguration",
        palette=palette,
        dodge=False,
        legend=False,
        width=0.62,
        linewidth=1.15,
        fliersize=4.5,
        ax=ax,
    )

    ax.set_title(
        title,
        pad=12,
    )

    ax.set_xlabel("Test-RMSE [K]")
    ax.set_ylabel("")

    ax.set_xlim(0.04, 0.72)

    ax.xaxis.set_major_formatter(
        FuncFormatter(comma_tick)
    )

    ax.grid(
        axis="y",
        visible=False,
    )

    ax.grid(
        axis="x",
        color="#D9D9D9",
        linewidth=0.8,
    )

    ax.spines[
        ["top", "right"]
    ].set_visible(False)

    ax.tick_params(
        axis="y",
        labelsize=12,
    )


fig.suptitle(
    (
        "Single-Step-Vorhersage: Einfluss des "
        "Physics-Loss und der Datenmenge"
    ),
    fontsize=20,
    fontweight="semibold",
    y=0.975,
)



fig.subplots_adjust(
    top=0.82,
    bottom=0.12,
    left=0.12,
    right=0.98,
    hspace=0.48,
    wspace=0.22,
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    OUTPUT,
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
)

print(
    f"\nGrafik gespeichert unter:\n{OUTPUT}"
)

plt.show()