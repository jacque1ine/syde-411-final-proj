"""Plot the selected layout, pooled trade-off and search histories."""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def plot_maps(grid):
    active_mask = grid.active_demand.to_numpy().reshape(15, 15)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.1))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.25, top=0.86, wspace=0.25)
    for ax, col, title, cmap in zip(
        axs,
        [
            "population_2021_sample_based_area_estimate",
            "recorded_fires_2018_2023",
            "risk_per_1000_over_6_years",
        ],
        [
            "Estimated population",
            "Recorded fires 2018–2023",
            "Risk and 8 selected stations",
        ],
        ["YlGnBu", "YlOrRd", "magma_r"],
    ):
        z = grid[col].to_numpy().reshape(15, 15)
        im = ax.imshow(
            np.ma.masked_where(~active_mask, z),
            origin="lower",
            extent=[0, 15, 0, 15],
            cmap=cmap,
            vmin=0,
        )
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    z = grid[grid.nsga2_station_selected]
    axs[2].scatter(
        z.column_from_west - 0.5,
        z.row_from_south - 0.5,
        c="#13d7e7",
        edgecolor="#17374a",
        s=80,
        marker="*",
    )
    fig.text(
        0.05,
        0.09,
        "Population: estimated residents  |  Fires: six-year counts  |  Risk: six-year fires per 1,000 estimated residents",
        fontsize=9,
    )
    fig.text(
        0.05,
        0.025,
        "Toronto study window: 225 cells, 210 included. Stars: NSGA-II result, seed 411. Blank cells: outside-city areas or a negligible boundary sliver.",
        fontsize=8,
    )
    fig.savefig(PROJECT_DIR / "opening_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_front(front, summary):
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(front.stations, front.risk_weighted_distance, "o-", ms=3, color="#087e8b")
    ax.scatter(
        [8],
        [summary["selected"]["risk_weighted_distance"]],
        color="#d14d2e",
        s=50,
        zorder=4,
        label="Mapped 8-station example",
    )
    ax.set_xlabel("Number of stations")
    ax.set_ylabel("Risk-weighted distance")
    ax.set_title("Pooled nondominated solutions from five runs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / "pareto.png", dpi=200)
    plt.close(fig)


def plot_convergence(histories):
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    for seed, history in histories.items():
        ax.plot(history.generation, history.hypervolume, label=str(seed), lw=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized hypervolume")
    ax.set_title("Search progress by random seed; higher is better")
    ax.legend(ncol=5, fontsize=8, loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / "convergence.png", dpi=200)
    plt.close(fig)


def main():
    grid = pd.read_csv(RESULTS_DIR / "selected_layout_225_cells.csv")
    summary = json.loads((RESULTS_DIR / "experiment_summary.json").read_text())
    front = pd.read_csv(RESULTS_DIR / "pooled_front.csv")
    histories = {
        seed: pd.read_csv(RESULTS_DIR / f"history_{seed}.csv")
        for seed in summary["configuration"]["seeds"]
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    plot_maps(grid)
    plot_front(front, summary)
    plot_convergence(histories)
    (PROJECT_DIR / "slide_data.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "front": front.to_dict("list"),
                "histories": {
                    str(seed): history.iloc[::10].to_dict("list")
                    for seed, history in histories.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
