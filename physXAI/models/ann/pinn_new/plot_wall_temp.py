"""Berechnete 2R2C-Wandtemperatur aus einer CSV plotten.

Direkter Aufruf aus dem physXAI-Hauptordner:

    python -m physXAI.models.ann.pinn_new.plot_wandtemperatur_2r2c \
        wandtemperatur_2r2c.csv --show
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pandas as pd

from physXAI.plotting.plotting import plot_wall_temperature


def main(
    result_path: Path,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> Path:
    """Liest die Ergebnis-CSV und speichert den Plot als HTML."""

    result_path = Path(result_path)

    if not result_path.exists():
        raise FileNotFoundError(
            f"Ergebnisdatei nicht gefunden: {result_path}"
        )

    result = pd.read_csv(result_path)

    figure = plot_wall_temperature(result)

    if output_path is None:
        output_path = result_path.with_suffix(".html")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        output_path,
        include_plotlyjs=True,
    )

    if show:
        figure.show()

    print(f"Plot: {output_path}")

    return output_path


def _parse_arguments():
    parser = ArgumentParser(
        description=(
            "Plottet gemessene Raumtemperatur und berechnete "
            "2R2C-Wandtemperatur."
        )
    )

    parser.add_argument(
        "result_path",
        type=Path,
        help=(
            "Pfad zur berechneten "
            "Wandtemperatur-CSV."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionaler Pfad der HTML-Datei.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Öffnet den Plot zusätzlich interaktiv.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    main(
        result_path=args.result_path,
        output_path=args.output,
        show=args.show,
    )