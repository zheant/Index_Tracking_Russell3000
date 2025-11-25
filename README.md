This repository contains my research project completed during my Summer 2025 internship at CIRRELT.

👉 [Internship Report (PDF)](https://github.com/aubejay22/Index_Tracking/raw/main/Recherche_ete25%20(4).pdf)

## Running the Russell 3000 pipeline

The methodology is unchanged from the S&P 500 experiments (50 exemplars using k-medoids vs. Gurobi); the defaults below only adjust the data source to the Russell 3000 and set the portfolio size to 300 exemplars.

1. **Create and activate a Python virtual environment, then install dependencies:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Normalise the raw constituent files** (reads `financial_data/russell3000/constituants_raw/*.csv`, writes to `financial_data/russell3000/constituants/` and aggregates `all_permnos.csv`):

   ```bash
   python scripts/prepare_russell_constituents.py
   ```

3. **Download Russell 3000 returns from WRDS** (uses the permno union from the previous step, outputs `returns_stocks.csv` and `returns_index.csv` under `financial_data/russell3000/`):

   ```bash
   python scripts/download_wrds_russell_data.py \
       --permno-csv financial_data/russell3000/constituants/all_permnos.csv \
       --start-date 2014-01-02 \
       --end-date 2023-12-31
   ```

   The script prompts for WRDS credentials and will reuse your configured ~/.pgpass if present.

4. **Run the optimisation pipeline** (defaults to Russell 3000, 300 exemplars, 3-year training window, yearly rebalancing). The flag `--replicator_cores` controls the OpenMP threads used by ReplicaTOR (8 on the c6i.2xlarge example below). The solver time limit is configurable and the distance metric now defaults to Pearson correlation:

   ```bash
   python main.py --solution_name quob --cardinality 300 --index russell3000 \
       --start_date 2014-01-02 --end_date 2023-12-31 --rebalancing 12 --T 3 \
       --replicator_cores 8 --time_limit 300 --distance_method pearson
   ```

   Swap `--solution_name gurobi` (or `quob_cor`, `gurobi_cor`, `lagrange_backward`, etc.) to compare optimisation approaches without changing the surrounding workflow.

   * `--time_limit` sets the maximum solve time (seconds) for both ReplicaTOR and Gurobi.
   * `--distance_method` chooses between distance correlation (`dcor`) and Pearson correlation (`pearson`) when building the distance matrix used by the solvers (default `pearson`).

   ReplicaTOR is expected at `~/or_tool/ReplicaTOR/cmake-build/ReplicaTOR`; adjust that path in `prafa/quob.py` if your binary lives elsewhere.

5. **Analyse Russell 3000 results (matches `analyses_resultats.ipynb`)**. After running QUBO/ReplicaTOR and/or Gurobi, generate the cumulative returns, tracking errors, absolute errors, and error distributions using the same methodology as the notebook:

   ```bash
   python scripts/analyze_results.py --index russell3000 --cardinality 300 \
       --solvers quob gurobi --output_dir results/analysis
   ```

   Adjust `--solvers` to `quob` or `gurobi` if you only want one method, and change `--result_path` if your portfolio pickles live elsewhere.
