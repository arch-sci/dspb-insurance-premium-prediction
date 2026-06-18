# Data Science Projects for Business — Fire Insurance Premium Prediction

Final project on a Crédit Agricole Assurances dataset, predicting the fire insurance premium for agricultural firms.

**Assignment:** full data science lifecycle on 166,943 contracts / 373 features with hidden column names — unsupervised exploration, two supervised prediction models, and a deployment discussion. Submitted as a report + competition predictions (RMSE-ranked) + notebook. Full details in the report.

**Context:** the assignment uses real, anonymized data from Crédit Agricole Assurances via [challengedata.ens.fr](https://challengedata.ens.fr), a data science challenge platform run by École Normale Supérieure (ENS Paris) in partnership with Collège de France and Institut Louis Bachelier — comparable to Kaggle, but sourcing challenges directly from companies and research labs. Predictions were ranked on the platform's public leaderboard via RMSE.

**Target:** `CHARGE = FREQ × CM × ANNEE_ASSURANCE` (claim frequency × average claim cost × years insured).

**Approach:**
- **Preprocessing:** detected a structural missing-data block (56-59% missing, ~0.92 mean pairwise correlation, 183 columns) → modeled as MNAR via a pattern-mixture approximation, encoded as a single binary legacy flag rather than dropping the columns.
- **Unsupervised (Phase 2):** PCA (~80 components for 80% variance) then K-means (k=4) on the PCA space to segment farms by physical scale, risk profile, and topography. Found one cluster (large, high-capital farms) with both the highest claim frequency and highest average cost, motivating separate frequency/severity modeling.
- **Supervised (Phase 3):** two XGBoost models (Poisson for FREQ, Gamma for CM, trained on claims only) rather than predicting CHARGE directly, with domain-aggregated features, exposure weighting, outlier capping, RandomizedSearchCV tuning, and 5-fold OOF cross-validation. Iterated through baselines, HistGradientBoosting, and a PyTorch MLP before settling on tuned XGBoost as the best and most explainable option.
- **Result:** test RMSE 5596.2 (best submission), with SHAP-based explainability for individual pricing decisions.
- **Deployment:** proposed as a real-time pricing API for new quotes, plus an overnight batch re-scoring job for renewal pricing and drift monitoring.

## Stack

Python · pandas · scikit-learn (PCA, KMeans, pipelines) · XGBoost · SHAP · matplotlib · seaborn · scipy
