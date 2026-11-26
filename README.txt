SYDE 411 PYTHON EXPERIMENT
Authors: Annie Cai, Michelle Wang, Jacqueline Fung

Install: python -m pip install -r requirements.txt
Run: python run_nsga2.py
Plot: python make_figures.py
Verify saved outputs: python verify_results.py

Use Python 3.13 for the recorded environment. The frozen Toronto CSV is the optimization input. Results/ holds measured outputs. Seeds are 411-415, population 120, generations 200. Original measured run times are in experiment_log.txt and experiment_summary.json; times change on rerun.

source_snapshot/ contains original data downloads, metadata, fingerprints and spatial preparation code. Its reproduce.py also produces a preliminary greedy station illustration; that is NOT the NSGA-II result used in the final report. The report uses results/selected_layout_225_cells.csv. Keep raw snapshots to reproduce the same instance.

Main conclusion: eight stations were the best NSGA-II count in these runs; an exact minimum-set-cover solve proved seven sufficient. No exact risk-distance frontier was computed. Coverage applies to cell centres within 4 km Euclidean distance.
