import argparse
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from scipy.stats import kurtosis, skew

from prafa.universe import Universe


DEFAULT_SOLVERS = ["quob", "gurobi"]


def build_universe(args: argparse.Namespace, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Universe:
    analysis_args = Namespace(
        index=args.index,
        data_path=args.data_path,
        result_path=args.result_path,
        solution_name="analysis",
        rebalancing=args.rebalancing,
        cardinality=args.cardinality,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    return Universe(analysis_args)


def load_portfolios(path: Path) -> Dict[pd.Timestamp, np.ndarray]:
    with path.open("rb") as f:
        portfolios = pickle.load(f)
    return portfolios


def resolve_portfolio_path(args: argparse.Namespace, solver: str) -> Path:
    filename = f"portfolio_{args.index}_{solver}_{args.cardinality}.json"
    return Path(args.result_path) / filename


def extract_timeseries(
    portfolios: Dict[pd.Timestamp, Iterable[float]],
    universe: Universe,
) -> Tuple[pd.Series, pd.Series, Dict[pd.Timestamp, float], Dict[pd.Timestamp, float]]:
    dates = sorted(portfolios.keys())
    if len(dates) < 2:
        raise ValueError("At least two rebalancing dates are required to compute out-of-sample metrics.")

    rendements_portefeuille: List[float] = []
    rendements_indice: List[float] = []
    index_dates: List[pd.Timestamp] = []
    tracking_errors: Dict[pd.Timestamp, float] = {}
    mae: Dict[pd.Timestamp, float] = {}

    for i in range(len(dates) - 1):
        start_date = pd.Timestamp(dates[i])
        end_date = pd.Timestamp(dates[i + 1]) - pd.tseries.offsets.BDay(1)
        universe.new_universe(start_date, end_date, training=False)
        X_test = universe.get_stocks_returns()
        Y_test = universe.get_index_returns()

        weights = portfolios[start_date]
        if isinstance(weights, dict):
            weight_vector = pd.Series(weights).reindex(X_test.columns).fillna(0).to_numpy()
        else:
            weight_vector = np.asarray(weights)
            if weight_vector.shape[0] != X_test.shape[1]:
                raise ValueError(
                    f"Weight length {weight_vector.shape[0]} does not match returns matrix width {X_test.shape[1]}"
                )

        return_outsample = X_test.to_numpy() @ weight_vector
        tracking_errors[X_test.index[-1]] = (return_outsample - Y_test.to_numpy()).std()
        mae[X_test.index[-1]] = np.abs(return_outsample - Y_test.to_numpy()).mean()

        rendements_portefeuille.extend(return_outsample.tolist())
        rendements_indice.extend(Y_test.tolist())
        index_dates.extend(list(X_test.index))

    rendements_portefeuille_series = pd.Series(rendements_portefeuille, index=index_dates)
    rendements_indice_series = pd.Series(rendements_indice, index=index_dates)
    return rendements_portefeuille_series, rendements_indice_series, tracking_errors, mae


def plot_cumulative_returns(rendements: Dict[str, pd.Series], indice_reference: pd.Series, output_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    for method, rp in rendements.items():
        plt.plot((rp + 1).cumprod(), label=f"Portefeuille – {method}")
    plt.plot((indice_reference + 1).cumprod(), label="Indice", color="black")
    plt.title("Rendements cumulés")
    plt.xlabel("Date")
    plt.ylabel("Rendement cumulé")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = output_dir / "cumulative_returns.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_tracking_errors(tracking_errors_all: Dict[str, Dict[pd.Timestamp, float]], output_dir: Path) -> Path:
    plt.figure(figsize=(10, 4))
    for method, te in tracking_errors_all.items():
        dates = list(te.keys())
        values = list(te.values())
        plt.scatter(dates, values, label=f"Tracking Error – {method}", s=25)
    plt.title("Tracking Errors")
    plt.xlabel("Date")
    plt.ylabel("Écart-type (Tracking Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = output_dir / "tracking_errors.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_absolute_errors(mae_all: Dict[str, Dict[pd.Timestamp, float]], output_dir: Path) -> Path:
    plt.figure(figsize=(10, 4))
    for method, ae in mae_all.items():
        dates = list(ae.keys())
        values = list(ae.values())
        plt.scatter(dates, values, label=f"Tracking Absolute Error – {method}", s=25)
    plt.title("Tracking Absolute Error")
    plt.xlabel("Date")
    plt.ylabel("Écart-type (Tracking Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = output_dir / "absolute_errors.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_cumulative_and_absolute(rendements: Dict[str, pd.Series], indice_reference: pd.Series, output_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    for method, rp in rendements.items():
        ax1.plot((rp + 1).cumprod(), label=f"Portefeuille – {method}")
    ax1.plot((indice_reference + 1).cumprod(), label="Indice", color="black")
    ax1.set_title("Rendements cumulés")
    ax1.set_ylabel("Rendement cumulé")
    ax1.legend()
    ax1.grid(True)

    for method, rp in rendements.items():
        ecarts_absolus = rp - indice_reference
        ax2.plot(rp.index, ecarts_absolus, label=f"Écarts absolus – {method}", alpha=0.5)
    ax2.set_title("Écarts absolus à chaque pas de temps")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Écart absolu")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    output_path = output_dir / "cumulative_and_absolute.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_error_distributions(rendements: Dict[str, pd.Series], indice_reference: pd.Series, output_dir: Path) -> Path:
    fig, axes = plt.subplots(nrows=len(rendements), ncols=1, figsize=(13, 5 * len(rendements)), sharex=True)
    if len(rendements) == 1:
        axes = [axes]

    for ax, (method, rp) in zip(axes, rendements.items()):
        erreurs = rp - indice_reference

        moyenne = erreurs.mean()
        mediane = erreurs.median()
        variance = erreurs.var()
        skewness = skew(erreurs)
        kurt = kurtosis(erreurs)

        sns.histplot(erreurs, kde=True, bins=200, ax=ax, color="skyblue", edgecolor="black")
        ax.axvline(moyenne, color="red", linestyle="--", linewidth=1.5, label=f"Moyenne: {moyenne:.5f}")
        ax.axvline(mediane, color="green", linestyle="--", linewidth=1.5, label=f"Médiane: {mediane:.5f}")

        ax.set_title(f"Distribution des erreurs de réplication – {method}", fontsize=12)
        ax.set_ylabel("Densité")
        ax.legend()

        textstr = "\n".join(
            [
                f"Variance : {variance:.6f}",
                f"Skewness : {skewness:.3f}",
                f"Kurtosis : {kurt:.3f}",
            ]
        )
        ax.text(
            0.98,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

    axes[-1].set_xlabel("Erreur de réplication (r_portefeuille - r_indice)")
    plt.tight_layout()
    output_path = output_dir / "error_distributions.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def analyze_results(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_paths = {}
    for solver in args.solvers:
        path = resolve_portfolio_path(args, solver)
        if not path.exists():
            raise FileNotFoundError(f"Portfolio file not found for {solver}: {path}")
        method_label = "QUBO" if solver.startswith("quob") else "Gurobi"
        method_paths[method_label] = path

    rendements: Dict[str, pd.Series] = {}
    tracking_errors_all: Dict[str, Dict[pd.Timestamp, float]] = {}
    mae_all: Dict[str, Dict[pd.Timestamp, float]] = {}
    indice_reference: pd.Series | None = None

    for method, path in method_paths.items():
        portfolios = load_portfolios(path)
        all_dates = sorted(portfolios.keys())
        start_date = pd.Timestamp(all_dates[0])
        end_date = pd.Timestamp(all_dates[-1]) + relativedelta(days=1)
        universe = build_universe(args, start_date, end_date)
        rp, ri, te, ae = extract_timeseries(portfolios, universe)
        rendements[method] = rp
        tracking_errors_all[method] = te
        mae_all[method] = ae
        if indice_reference is None:
            indice_reference = ri

    assert indice_reference is not None, "No index reference series was created."

    outputs = {
        "cumulative": plot_cumulative_returns(rendements, indice_reference, output_dir),
        "tracking_error": plot_tracking_errors(tracking_errors_all, output_dir),
        "absolute_error": plot_absolute_errors(mae_all, output_dir),
        "cumulative_and_absolute": plot_cumulative_and_absolute(rendements, indice_reference, output_dir),
        "error_distributions": plot_error_distributions(rendements, indice_reference, output_dir),
    }

    for name, path in outputs.items():
        print(f"Saved {name} plot to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse les résultats Russell 3000 (QUBO/Gurobi)")
    parser.add_argument("--data_path", type=str, default="financial_data", help="Chemin des données")
    parser.add_argument("--result_path", type=str, default="results", help="Chemin des résultats")
    parser.add_argument("--index", type=str, default="russell3000", help="Indice à analyser")
    parser.add_argument("--cardinality", type=int, default=300, help="Cardinalité du portefeuille")
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=DEFAULT_SOLVERS,
        default=DEFAULT_SOLVERS,
        help="Choisir les résultats à analyser (quob, gurobi)",
    )
    parser.add_argument("--rebalancing", type=int, default=12, help="Rebalancement (mois) utilisé lors de l'optimisation")
    parser.add_argument("--output_dir", type=str, default="results/analysis", help="Dossier de sortie pour les graphiques")
    return parser.parse_args()


if __name__ == "__main__":
    analyze_results(parse_args())
