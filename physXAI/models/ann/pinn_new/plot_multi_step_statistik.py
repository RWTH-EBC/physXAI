from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


BASE_DIR = Path(
    r"D:\phe-dwe\Git\testhall_offices_experiment"
    r"\0_agentlib_configs\results\sweep_multi_step"
)


DATASET_FOLDERS = {
    "W28T": BASE_DIR / "W28T",
    "W14T": BASE_DIR / "W14T",
}

DATASET_TITLES = {
    "W28T": "28 Tage",
    "W14T": "14 Tage",
}

OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PDF = True

EXTREME_RMSE_LIMIT = 5.0

PINN_MODELS = [
    "1R1C",
    "2R2C",
    "Gokhale",
]

MODEL_ORDER = [
    "ANN",
    "1R1C",
    "2R2C",
    "Gokhale",
]

COLORS = {
    "ANN": "#222222",
    "1R1C": "#0072B2",
    "2R2C": "#D55E00",
    "Gokhale": "#009E73",
}

HORIZON_LABELS = [
    "0,5",
    "1",
    "3",
    "6",
    "12",
    "20",
    "23",
]

HORIZON_COLUMNS = [
    "test_rmse_30min",
    "test_rmse_1h",
    "test_rmse_3h",
    "test_rmse_6h",
    "test_rmse_12h",
    "test_rmse_20h",
    "test_rmse_full_horizon",
]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def find_csv(dataset: str, file_type: str) -> Path:

    dataset_folder = DATASET_FOLDERS[dataset]
    dataset_lower = dataset.lower()

    candidates = [
        dataset_folder / f"multi_step_{file_type}.csv",
        dataset_folder / f"multi_step_{file_type}_{dataset_lower}.csv",
        BASE_DIR / f"multi_step_{file_type}_{dataset_lower}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched_paths = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"Keine Datei für {dataset}/{file_type} gefunden.\n"
        f"Gesucht wurde:\n{searched_paths}"
    )


def read_failures(path: Path) -> pd.DataFrame:

    try:
        return pd.read_csv(path)

    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "model",
                "lambda",
                "run",
            ]
        )


def load_data():

    results = {}
    failures = {}

    for dataset in DATASET_FOLDERS:
        results_path = find_csv(
            dataset=dataset,
            file_type="results",
        )

        failures_path = find_csv(
            dataset=dataset,
            file_type="failures",
        )

        results[dataset] = pd.read_csv(results_path)
        failures[dataset] = read_failures(failures_path)

        print(f"{dataset} Ergebnisse: {results_path}")
        print(f"{dataset} Fehlläufe:  {failures_path}")

    return results, failures


def select_positive_lambdas(
    w28_results: pd.DataFrame,
) -> dict[str, float | None]:

    selected_lambdas = {
        "ANN": None,
    }

    for model in PINN_MODELS:
        model_data = w28_results[
            w28_results["model"].eq(model)
            & w28_results["lambda"].gt(0)
        ]

        lambda_medians = (
            model_data
            .groupby("lambda")["val_rmse_full_horizon"]
            .median()
        )

        if lambda_medians.empty:
            raise ValueError(
                f"Für {model} wurden keine positiven Lambda-Werte gefunden."
            )

        best_lambda = lambda_medians.idxmin()

        selected_lambdas[model] = float(best_lambda)

    return selected_lambdas


def selected_group(
    data: pd.DataFrame,
    model: str,
    selected_lambdas: dict[str, float | None],
) -> pd.DataFrame:

    if model == "ANN":
        return data[
            data["model"].eq("ANN")
        ]

    selected_lambda = selected_lambdas[model]

    return data[
        data["model"].eq(model)
        & np.isclose(
            data["lambda"],
            selected_lambda,
            equal_nan=False,
        )
    ]


def format_lambda(value: float | None) -> str:

    if value is None:
        return ""

    return f"{value:g}".replace(".", ",")


def model_label(
    model: str,
    selected_lambdas: dict[str, float | None],
) -> str:

    if model == "ANN":
        return "ANN"

    selected_lambda = selected_lambdas[model]

    return (
        rf"{model} "
        rf"($\lambda$ = {format_lambda(selected_lambda)})"
    )



def decimal_comma(value, position=None):

    return f"{value:g}".replace(".", ",")


def style_axes(ax: plt.Axes):

    ax.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.8,
        alpha=0.75,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        labelsize=10,
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
):

    png_path = OUTPUT_DIR / f"{filename}.png"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Gespeichert: {png_path}")

    if SAVE_PDF:
        pdf_path = OUTPUT_DIR / f"{filename}.pdf"

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        print(f"Gespeichert: {pdf_path}")

    plt.close(fig)



def plot_horizon_rmse(
    results: dict[str, pd.DataFrame],
    selected_lambdas: dict[str, float | None],
):


    positions = np.arange(
        len(HORIZON_COLUMNS)
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14.5, 5.7),
        sharey=True,
    )

    for ax, dataset in zip(
        axes,
        DATASET_FOLDERS,
    ):
        data = results[dataset]

        for model in MODEL_ORDER:
            group = selected_group(
                data=data,
                model=model,
                selected_lambdas=selected_lambdas,
            )

            median = np.array(
                [
                    group[column].median()
                    for column in HORIZON_COLUMNS
                ]
            )

            q25 = np.array(
                [
                    group[column].quantile(0.25)
                    for column in HORIZON_COLUMNS
                ]
            )

            q75 = np.array(
                [
                    group[column].quantile(0.75)
                    for column in HORIZON_COLUMNS
                ]
            )

            ax.plot(
                positions,
                median,
                color=COLORS[model],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                label=model_label(
                    model,
                    selected_lambdas,
                ),
                zorder=3,
            )

            ax.fill_between(
                positions,
                q25,
                q75,
                color=COLORS[model],
                alpha=0.14,
                linewidth=0,
            )

        ax.set_title(
            DATASET_TITLES[dataset],
            pad=11,
        )

        ax.set_xlabel(
            "Vorhersagehorizont in h"
        )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            HORIZON_LABELS
        )

        ax.set_xlim(
            -0.15,
            len(positions) - 0.85,
        )

        ax.set_ylim(
            0,
            2.05,
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(decimal_comma)
        )

        style_axes(ax)

    axes[0].set_ylabel(
        "Kumulativer Test-RMSE in K"
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        "Multi-Step-Vorhersagefehler "
        "über den Vorhersagehorizont",
        fontsize=15,
        fontweight="bold",
        y=1.055,
    )

    fig.text(
        0.5,
        0.015,
        "Linie: Median · "
        "Farbband: Interquartilsabstand · "
        "λ-Auswahl anhand der W28T-Validierung",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 0.92)
    )

    save_figure(
        fig=fig,
        filename="01_rmse_ueber_horizont",
    )


def plot_full_horizon(
    results: dict[str, pd.DataFrame],
    selected_lambdas: dict[str, float | None],
):
    """
    Dot-Whisker-Plot für den Full-Horizon-Test-RMSE.

    Punkt:
        Median

    Breite Linie:
        Interquartilsabstand
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.5, 5.4),
        sharex=True,
    )

    plot_order = MODEL_ORDER[::-1]

    for ax, dataset in zip(
        axes,
        DATASET_FOLDERS,
    ):
        data = results[dataset]

        for y_position, model in enumerate(
            plot_order
        ):
            group = selected_group(
                data=data,
                model=model,
                selected_lambdas=selected_lambdas,
            )

            values = group[
                "test_rmse_full_horizon"
            ]

            median = values.median()
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)

            ax.hlines(
                y=y_position,
                xmin=q25,
                xmax=q75,
                color=COLORS[model],
                linewidth=5,
                alpha=0.35,
                zorder=2,
            )

            ax.plot(
                median,
                y_position,
                "o",
                color=COLORS[model],
                markersize=9,
                zorder=3,
            )

            ax.text(
                q75 + 0.035,
                y_position,
                f"{median:.3f}".replace(".", ","),
                va="center",
                fontsize=9.5,
            )

        ax.set_yticks(
            range(len(plot_order))
        )

        ax.set_yticklabels(
            [
                model_label(
                    model,
                    selected_lambdas,
                )
                for model in plot_order
            ]
        )

        ax.set_title(
            DATASET_TITLES[dataset],
            pad=12,
        )

        ax.set_xlabel(
            "Test-RMSE über 23 h in K"
        )

        ax.set_xlim(
            0,
            2.05,
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(decimal_comma)
        )

        ax.grid(
            axis="x",
            color="#D9D9D9",
            linewidth=0.8,
            alpha=0.75,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Full-Horizon-Modellvergleich",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        0.015,
        "Punkt: Median · "
        "breite Linie: Interquartilsabstand · "
        "kleinere Werte sind besser",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 0.96)
    )

    save_figure(
        fig=fig,
        filename="02_full_horizon_vergleich",
    )


def plot_6h_horizon(
    results: dict[str, pd.DataFrame],
    selected_lambdas: dict[str, float | None],
):

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.5, 5.4),
        sharex=True,
    )

    plot_order = MODEL_ORDER[::-1]

    for ax, dataset in zip(
        axes,
        DATASET_FOLDERS,
    ):
        data = results[dataset]

        for y_position, model in enumerate(plot_order):
            group = selected_group(
                data=data,
                model=model,
                selected_lambdas=selected_lambdas,
            )

            values = group["test_rmse_6h"]

            median = values.median()
            q25 = values.quantile(0.25)
            q75 = values.quantile(0.75)

            ax.hlines(
                y=y_position,
                xmin=q25,
                xmax=q75,
                color=COLORS[model],
                linewidth=5,
                alpha=0.35,
                zorder=2,
            )

            ax.plot(
                median,
                y_position,
                "o",
                color=COLORS[model],
                markersize=9,
                zorder=3,
            )

            ax.text(
                q75 + 0.035,
                y_position,
                f"{median:.3f}".replace(".", ","),
                va="center",
                fontsize=9.5,
            )

        ax.set_yticks(
            range(len(plot_order))
        )

        ax.set_yticklabels(
            [
                model_label(
                    model,
                    selected_lambdas,
                )
                for model in plot_order
            ]
        )

        ax.set_title(
            DATASET_TITLES[dataset],
            pad=12,
        )

        ax.set_xlabel(
            "Kumulativer Test-RMSE über 6 h in K"
        )

        ax.set_xlim(
            0,
            2.05,
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(decimal_comma)
        )

        ax.grid(
            axis="x",
            color="#D9D9D9",
            linewidth=0.8,
            alpha=0.75,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "6-Stunden-Modellvergleich",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    fig.text(
        0.5,
        0.015,
        "Punkt: Median · "
        "breite Linie: Interquartilsabstand · "
        "kleinere Werte sind besser",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=(0, 0.05, 1, 0.96)
    )

    save_figure(
        fig=fig,
        filename="02b_6h_horizon_vergleich",
    )

def plot_lambda_sensitivity(
    results: dict[str, pd.DataFrame],
):

    data = results["W28T"]

    lambdas = sorted(
        value
        for value in data["lambda"].dropna().unique()
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15.2, 5.3),
    )

    for ax, model in zip(
        axes,
        PINN_MODELS,
    ):
        medians = []
        q25_values = []
        q75_values = []
        counts = []

        for lambda_value in lambdas:
            group = data[
                data["model"].eq(model)
                & np.isclose(
                    data["lambda"],
                    lambda_value,
                    equal_nan=False,
                )
            ]

            values = group[
                "val_rmse_full_horizon"
            ]

            medians.append(
                values.median()
            )

            q25_values.append(
                values.quantile(0.25)
            )

            q75_values.append(
                values.quantile(0.75)
            )

            counts.append(
                len(group)
            )

        medians = np.asarray(medians)
        q25_values = np.asarray(q25_values)
        q75_values = np.asarray(q75_values)

        positions = np.arange(
            len(lambdas)
        )

        lower_error = np.maximum(
            medians - q25_values,
            1e-12,
        )

        upper_error = np.maximum(
            q75_values - medians,
            1e-12,
        )

        ax.errorbar(
            positions,
            medians,
            yerr=[
                lower_error,
                upper_error,
            ],
            fmt="o-",
            color=COLORS[model],
            linewidth=2,
            markersize=6,
            capsize=4,
            elinewidth=1.7,
        )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            [
                format_lambda(value)
                for value in lambdas
            ]
        )

        ax.set_xlabel(
            r"$\lambda$"
        )

        ax.set_title(
            model
        )

        ax.set_yscale(
            "log"
        )

        ax.yaxis.set_major_locator(
            LogLocator(
                base=10,
                numticks=8,
            )
        )

        ax.yaxis.set_minor_formatter(
            NullFormatter()
        )

        ax.grid(
            axis="y",
            which="major",
            color="#D9D9D9",
            linewidth=0.8,
            alpha=0.75,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Fehlende Läufe direkt markieren.
        for x_position, (median, count) in enumerate(
            zip(medians, counts)
        ):
            if count < 20:
                ax.annotate(
                    f"n={count}",
                    (x_position, median),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#A33A2B",
                )

    axes[0].set_ylabel(
        "Full-Horizon-Validierungs-RMSE in K (log.)"
    )

    fig.suptitle(
        "Einfluss der Physics-Loss-Gewichtung – 28 Tage",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    fig.text(
        0.5,
        0.015,
        "Punkt: Median · "
        "Fehlerbalken: Interquartilsabstand · "
        "n < 20 kennzeichnet harte Fehlläufe",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=(0, 0.06, 1, 0.96)
    )

    save_figure(
        fig=fig,
        filename="03_lambda_sensitivitaet",
    )


# =============================================================================
# Abbildung 4:
# Numerische Stabilität
# =============================================================================

def count_failures(
    failure_data: pd.DataFrame,
    model: str,
    selected_lambdas: dict[str, float | None],
) -> int:

    model_failures = failure_data[
        failure_data["model"].eq(model)
    ]

    if model_failures.empty:
        return 0

    if model == "ANN":
        return len(model_failures)

    selected_lambda = selected_lambdas[model]

    lambda_matches = np.isclose(
        model_failures["lambda"],
        selected_lambda,
        equal_nan=False,
    )

    return int(
        lambda_matches.sum()
    )


def plot_training_stability(
    results: dict[str, pd.DataFrame],
    failures: dict[str, pd.DataFrame],
    selected_lambdas: dict[str, float | None],
):

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.5, 5.5),
        sharey=True,
    )

    for ax, dataset in zip(
        axes,
        DATASET_FOLDERS,
    ):
        data = results[dataset]
        failure_data = failures[dataset]

        normal_runs = []
        extreme_runs = []
        failed_runs = []

        for model in MODEL_ORDER:
            group = selected_group(
                data=data,
                model=model,
                selected_lambdas=selected_lambdas,
            )

            normal_runs.append(
                int(
                    (
                        group["test_rmse_full_horizon"]
                        <= EXTREME_RMSE_LIMIT
                    ).sum()
                )
            )

            extreme_runs.append(
                int(
                    (
                        group["test_rmse_full_horizon"]
                        > EXTREME_RMSE_LIMIT
                    ).sum()
                )
            )

            failed_runs.append(
                count_failures(
                    failure_data=failure_data,
                    model=model,
                    selected_lambdas=selected_lambdas,
                )
            )

        positions = np.arange(
            len(MODEL_ORDER)
        )

        normal_array = np.asarray(
            normal_runs
        )

        extreme_array = np.asarray(
            extreme_runs
        )

        ax.bar(
            positions,
            normal_runs,
            color="#4C9F70",
            label=(
                f"RMSE ≤ "
                f"{EXTREME_RMSE_LIMIT:g} K"
            ),
        )

        ax.bar(
            positions,
            extreme_runs,
            bottom=normal_array,
            color="#E6A23C",
            label=(
                f"RMSE > "
                f"{EXTREME_RMSE_LIMIT:g} K"
            ),
        )

        ax.bar(
            positions,
            failed_runs,
            bottom=normal_array + extreme_array,
            color="#C84B4B",
            label="NaN-Fehllauf",
        )

        for index, (
            normal,
            extreme,
            failed,
        ) in enumerate(
            zip(
                normal_runs,
                extreme_runs,
                failed_runs,
            )
        ):
            if normal > 0:
                ax.text(
                    index,
                    normal / 2,
                    str(normal),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

            if extreme > 0:
                ax.text(
                    index,
                    normal + extreme / 2,
                    str(extreme),
                    ha="center",
                    va="center",
                    color="#333333",
                    fontweight="bold",
                )

            if failed > 0:
                ax.text(
                    index,
                    normal + extreme + failed / 2,
                    str(failed),
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                )

        ax.set_xticks(
            positions
        )

        ax.set_xticklabels(
            MODEL_ORDER
        )

        ax.set_ylim(
            0,
            20.8,
        )

        ax.set_yticks(
            [0, 5, 10, 15, 20]
        )

        ax.set_title(
            DATASET_TITLES[dataset],
            pad=12,
        )

        ax.set_xlabel(
            "Ausgewählte Konfiguration"
        )

        style_axes(ax)

    axes[0].set_ylabel(
        "Anzahl Trainingsläufe"
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        "Numerische Stabilität der ausgewählten Modelle",
        fontsize=15,
        fontweight="bold",
        y=1.07,
    )

    fig.text(
        0.5,
        0.015,
        "Je Konfiguration wurden 20 Läufe gestartet; "
        "extreme endliche Fehler werden separat ausgewiesen.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    fig.tight_layout(
        rect=(0, 0.06, 1, 0.92)
    )

    save_figure(
        fig=fig,
        filename="04_trainingsstabilitaet",
    )


def main():
    results, failures = load_data()

    selected_lambdas = select_positive_lambdas(
        results["W28T"]
    )

    print("\nAusgewählte Lambda-Werte:")

    for model in PINN_MODELS:
        print(
            f"  {model}: "
            f"{selected_lambdas[model]:g}"
        )

    print()

    plot_horizon_rmse(
        results=results,
        selected_lambdas=selected_lambdas,
    )

    plot_full_horizon(
        results=results,
        selected_lambdas=selected_lambdas,
    )

    plot_6h_horizon(
        results=results,
        selected_lambdas=selected_lambdas,
    )

    plot_lambda_sensitivity(
        results=results,
    )

    plot_training_stability(
        results=results,
        failures=failures,
        selected_lambdas=selected_lambdas,
    )

    print(
        f"\nAlle Abbildungen wurden gespeichert unter:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()