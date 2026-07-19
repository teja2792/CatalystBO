"""
explain_surrogate.py

Phase 5: explain what an interpretable surrogate model learns from the
(candidate, activity) pairs the Bayesian optimizer evaluated.

Note: the model actually driving BO's acquisition function is a Gaussian
Process, fit internally by skopt on a transformed representation of the
search space. GPs have no efficient exact SHAP explainer. Instead, this
fits a standalone RandomForestRegressor -- same preprocessing pattern as
CatalystML/ExplainableCatML -- on the (candidate, activity) pairs evaluated
across the 10 BO seeds, and explains THAT with SHAP's exact TreeExplainer.
This is a post-hoc interpretation of the optimizer's evidence, not a
reconstruction of the GP's internal beliefs.

A held-out test split reports a real generalization R^2, since a 300-tree
RF can nearly memorize 500 points / 6 features on a training-only score.
"""

import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

CATEGORICAL_FEATURES = ["composition", "facet", "defect_density"]
NUMERICAL_FEATURES = ["particle_size_nm", "metal_loading_wt_pct", "calcination_temp_C"]

df = pd.read_csv("results/multiseed_log.csv")
bo_df = df[df["method"] == "bayesian_optimization"].reset_index(drop=True)

X = bo_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
y = bo_df["activity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ("num", "passthrough", NUMERICAL_FEATURES),
])

model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=0)),
])
model.fit(X_train, y_train)

print(f"Surrogate trained on {len(X_train)} points, held out {len(X_test)} for testing.")
print(f"Train R^2: {model.score(X_train, y_train):.3f} | "
      f"Test R^2: {model.score(X_test, y_test):.3f}")

# Explain over the full evaluated dataset (500 points) for the most
# representative SHAP plot, even though the model was fit on 80% of it.
X_transformed = model.named_steps["preprocess"].transform(X)
feature_names = model.named_steps["preprocess"].get_feature_names_out()

explainer = shap.TreeExplainer(model.named_steps["rf"])
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                   plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("results/surrogate_shap_bar.png", dpi=150)
plt.close()

shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig("results/surrogate_shap_beeswarm.png", dpi=150)
plt.close()

print("Saved results/surrogate_shap_bar.png and results/surrogate_shap_beeswarm.png")