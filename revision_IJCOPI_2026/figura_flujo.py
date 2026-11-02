import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
steps = [
 ("Statistics V5 dataset\n3,049 records, 8 predictors, target: total catch", "#e8eef6"),
 ("For each replicate r = 1..60 (seed 1000 + r)\nand training proportion p ∈ {20%, …, 80%}", "#f3f3f3"),
 ("Random train/test split (same split for RF, BRT,\nraw and Min-Max versions → paired observations)", "#f3f3f3"),
 ("Optional Min-Max scaling\n(fitted on the training set only)", "#f3f3f3"),
 ("Hyperparameter search on the training set only\nRandomizedSearchCV: 15 configurations × 5-fold CV (MSE)", "#fdeee5"),
 ("Refit best RF and best XGBoost on the full training set", "#fdeee5"),
 ("Predict the untouched test set → MAE, RMSE, R²", "#f3f3f3"),
 ("Paired statistics over 60 replicates\nWilcoxon signed-rank, Holm correction, effect size, 95% CI", "#e8eef6"),
]
fig, ax = plt.subplots(figsize=(5.2, 7.2)); ax.axis("off")
h, gap, y = 0.085, 0.035, 0.97
for i, (t, c) in enumerate(steps):
    ax.add_patch(FancyBboxPatch((0.05, y-h), 0.9, h, boxstyle="round,pad=0.008", fc=c, ec="#555", lw=0.8))
    ax.text(0.5, y-h/2, t, ha="center", va="center", fontsize=8)
    if i < len(steps)-1:
        ax.annotate("", xy=(0.5, y-h-gap+0.004), xytext=(0.5, y-h-0.004), arrowprops=dict(arrowstyle="->", lw=0.9))
    y -= h+gap
ax.annotate("", xy=(0.97, 0.97-h/2-(h+gap)), xytext=(0.97, 0.97-h/2-6*(h+gap)),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#888", connectionstyle="arc3,rad=0"))
ax.text(0.985, 0.97-h/2-3.5*(h+gap), "next r, p", rotation=90, fontsize=7, color="#666", va="center")
ax.set_xlim(0,1.02); ax.set_ylim(y, 1)
fig.savefig("salidas/fig_flujo.png", dpi=300, bbox_inches="tight")
