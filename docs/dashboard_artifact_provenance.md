# Dashboard Artifact Provenance

This note records how dashboard values are sourced. Dashboard integration does not fit or retrain models.

## Member 2 evaluation

- `holdout_model_comparison.csv` is checked against the saved results table in cell 18 of `notebooks/Model_Training_Rating_Classification.ipynb`.
- When present in the integrated CSV, training times are checked against cell 16's saved output, and hyperparameter/balancing labels are checked against cell 16's training code.
- `loao_results.csv` is checked against the saved results table in cell 20 of that notebook.
- `confusion_matrices.json` contains the counts shown in all three cell 18 confusion-matrix plots; each displayed count was cross-checked against the saved notebook images. The read-only validator checks that their support and derived aggregate metrics match the saved holdout results table.
- The dashboard calculates class precision, recall, F1, and support from those confusion-matrix counts at load time. The rounded `classification_reports.json` file is retained as a legacy artifact but is no longer used by the dashboard.
- `python -m src.utils.export_artifacts` validates existing files only. It does not create metrics, fill in defaults, or train models.

## Inference bundle

The notebook's save cell writes `rating_classifier_bundle.pkl` under its relative `models/` directory. The saved notebook execution output resolved that directory to `notebooks/models/`, but that direct notebook-export file is not present in this workspace. The dashboard now checks `notebooks/models/rating_classifier_bundle.pkl` first and falls back to the existing `models/rating_classifier_bundle.pkl` so the predictor remains usable. The available root bundle's model type, TF-IDF parameters, model parameters, labels, and holdout metrics match the notebook configuration and saved holdout output; file lineage itself cannot be established from the available files.

To establish direct-export lineage, obtain Member 2's existing `notebooks/models/rating_classifier_bundle.pkl` from the original notebook run. No retraining is needed if that saved artifact is available.

## Preprocessing parity

The predictor's cleaning function was compared with the implementation in `notebooks/preprocessing.ipynb`. The 112-entry slang dictionary and text transformation steps match. The processed dataset was also checked row by row: all 609,900 `content_clean` values match the original notebook function (zero differences). This comparison does not modify the processed dataset.
