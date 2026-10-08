# User guide

Use `N_gonorrhoeae_AMR_reproducible_analysis.ipynb` as the primary notebook. `BIF_COMP_PROJECT.ipynb` is the preserved original submission and is retained for provenance rather than as the recommended execution path.

## Run the project

1. Create and activate a Python 3.13 virtual environment.
2. Install the pinned packages with `python -m pip install -r requirements.txt`.
3. Start Jupyter with `python -m jupyter lab`.
4. Open `N_gonorrhoeae_AMR_reproducible_analysis.ipynb`.
5. Choose **Restart Kernel and Run All Cells**.

The notebook automatically locates `DATA/metadata.csv`, validates the three unitig files, and recreates the tables and figures under `results/`.

## What to expect

- Azithromycin and ciprofloxacin are evaluated with grouped five-fold cross-validation.
- Cefixime is audited but not modelled because only five resistant isolates are available.
- Selected-model uncertainty is estimated with a 500-resample profile-group bootstrap.
- Continent-level results are descriptive slices of the out-of-fold predictions, not geographic holdout validation.
- The final cell writes `results/run_completion.json` only after the expected outputs exist.

## Common problems

- **Missing package:** confirm that the virtual environment is active and reinstall `requirements.txt`.
- **Missing data:** confirm that `DATA/` contains `metadata.csv` and all three `*_gwas_filtered_unitigs.Rtab` files.
- **Wrong notebook:** use the reproducible notebook for the verified result; the original submission is included only to document the project's development.
- **Slow execution:** the ciprofloxacin matrix has 8,873 unitigs and the notebook includes grouped bootstrap resampling. A complete run can take several minutes depending on the computer.

For the research question, reproduced metrics, limitations, and interpretation guidance, read `README.md`.
