"""
Análisis estadístico de resultados.csv (generado por experimento.py).
Salidas en ./salidas/: tablas CSV/Markdown y figuras PNG para el manuscrito revisado.
"""
import json, os, sys
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "salidas")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "resultados.csv"))
# solo réplicas completas
full = df.groupby("rep").size()
df = df[df.rep.isin(full[full == full.max()].index)]
R = df.rep.nunique()
PCTS = sorted(df.pct.unique())
rng = np.random.default_rng(0)
md = []


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj


def paired(a, b):
    """a, b alineados por réplica. Devuelve estadístico, p, r_rb, mediana dif, IC95 bootstrap."""
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return dict(W=np.nan, p=1.0, r_rb=0.0, med_diff=0.0, ci_lo=0.0, ci_hi=0.0, n=len(d))
    nz = d[d != 0]
    method = "exact" if len(nz) <= 50 and len(np.unique(np.abs(nz))) == len(nz) else "approx"
    res = stats.wilcoxon(a, b, zero_method="wilcox", method=method)
    ranks = stats.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    boots = np.median(rng.choice(d, (10000, len(d)), replace=True), axis=1)
    return dict(W=res.statistic, p=res.pvalue, r_rb=(wp - wm) / (wp + wm), med_diff=np.median(d),
                ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), n=len(d))


def wide(model, scaling, metric):
    s = df[(df.model == model) & (df.scaling == scaling)]
    return s.pivot(index="rep", columns="pct", values=metric)


# 1) Resumen descriptivo
desc = (df.groupby(["scaling", "model", "pct"])[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"]).round(3))
desc.to_csv(os.path.join(OUT, "tabla_descriptiva.csv"))
md.append(f"# Resultados (R = {R} réplicas)\n\n## Media ± DE por modelo, escalado y % de entrenamiento\n")
for sc in ["raw", "minmax"]:
    md.append(f"\n### Escalado: {sc}\n\n| % train | " + " | ".join(
        f"{m} {k}" for m in ["RF", "BRT", "Mean"] for k in ["MAE", "RMSE", "R²"]) + " |")
    md.append("|" + "---|" * 10)
    for p in PCTS:
        row = [f"{int(p*100)}"]
        for m in ["RF", "BRT", "Mean"]:
            for k in ["MAE", "RMSE", "R2"]:
                v = df[(df.scaling == sc) & (df.model == m) & (df.pct == p)][k]
                row.append(f"{v.mean():.2f} ± {v.std():.2f}")
        md.append("| " + " | ".join(row) + " |")

# 2) RF vs BRT por % (pareado por réplica)
rows = []
for sc in ["raw", "minmax"]:
    for metric in ["MAE", "RMSE"]:
        a, b = wide("RF", sc, metric), wide("BRT", sc, metric)
        tmp = [dict(scaling=sc, metric=metric, pct=p, **paired(a[p], b[p])) for p in PCTS]
        adj = holm([t["p"] for t in tmp])
        for t, q in zip(tmp, adj): t["p_holm"] = q
        rows += tmp
rfbrt = pd.DataFrame(rows)
rfbrt.to_csv(os.path.join(OUT, "wilcoxon_RF_vs_BRT.csv"), index=False)

# 3) raw vs minmax por modelo y %
rows = []
for m in ["RF", "BRT"]:
    for metric in ["MAE", "RMSE"]:
        a, b = wide(m, "raw", metric), wide(m, "minmax", metric)
        tmp = [dict(model=m, metric=metric, pct=p, **paired(a[p], b[p])) for p in PCTS]
        adj = holm([t["p"] for t in tmp])
        for t, q in zip(tmp, adj): t["p_holm"] = q
        rows += tmp
norm = pd.DataFrame(rows)
norm.to_csv(os.path.join(OUT, "wilcoxon_raw_vs_minmax.csv"), index=False)


def fmt_p(p):
    return "—" if np.isnan(p) else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")


for title, t, key in [("RF − BRT (diferencia pareada; negativo = RF mejor)", rfbrt, "scaling"),
                      ("Sin normalizar − Min-Max (negativo = sin normalizar mejor)", norm, "model")]:
    md.append(f"\n## Wilcoxon: {title}\n\n| {key} | métrica | % | mediana dif [IC95] | W | p | p Holm | r_rb |\n|---|---|---|---|---|---|---|---|")
    for _, w in t.iterrows():
        md.append(f"| {w[key]} | {w.metric} | {int(w.pct*100)} | {w.med_diff:.3f} [{w.ci_lo:.3f}, {w.ci_hi:.3f}] | "
                  f"{w.W if np.isnan(w.W) else int(w.W)} | {fmt_p(w.p)} | {fmt_p(w.p_holm)} | {w.r_rb:.2f} |")

# 4) Efecto del % de entrenamiento: Friedman + Wilcoxon por pares (Holm)
md.append("\n## Efecto del porcentaje de entrenamiento (datos sin normalizar)\n")
pmats = {}
for m in ["RF", "BRT"]:
    for metric in ["MAE", "RMSE"]:
        w = wide(m, "raw", metric)
        fr = stats.friedmanchisquare(*[w[p] for p in PCTS])
        pairs = list(combinations(PCTS, 2))
        ps = [stats.wilcoxon(w[i], w[j]).pvalue for i, j in pairs]
        adj = holm(ps)
        M = pd.DataFrame(np.nan, index=[int(p*100) for p in PCTS], columns=[int(p*100) for p in PCTS])
        for (i, j), q in zip(pairs, adj):
            M.loc[int(i*100), int(j*100)] = M.loc[int(j*100), int(i*100)] = q
        pmats[(m, metric)] = M
        M.to_csv(os.path.join(OUT, f"pmatrix_{m}_{metric}.csv"))
        md.append(f"\n### {m} – {metric}: Friedman χ² = {fr.statistic:.2f}, p = {fmt_p(fr.pvalue)}\n")
        md.append("p ajustados (Holm), Wilcoxon por pares:\n\n" + M.map(lambda v: "" if np.isnan(v) else fmt_p(v)).to_markdown())

# 5) Hiperparámetros seleccionados
md.append("\n## Hiperparámetros seleccionados (frecuencia en todas las corridas)\n")
for m in ["RF", "BRT"]:
    bp = df[df.model == m].best_params.map(json.loads)
    md.append(f"\n### {m}\n")
    for k in bp.iloc[0].keys():
        c = Counter(str(x[k]) for x in bp)
        md.append(f"- {k}: " + ", ".join(f"{v} ({n/len(bp)*100:.0f} %)" for v, n in c.most_common()))
md.append(f"\nTiempo medio por búsqueda: RF {df[df.model=='RF'].seconds.mean():.1f}s, BRT {df[df.model=='BRT'].seconds.mean():.1f}s")

open(os.path.join(OUT, "resumen.md"), "w").write("\n".join(md))

# 6) Figura compacta: MAE y RMSE vs % entrenamiento (media ± IC95), sin normalizar
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
col = {"RF": "#1f5f99", "BRT": "#d9622b"}
for ax, metric in zip(axes, ["MAE", "RMSE"]):
    for m in ["RF", "BRT"]:
        w = wide(m, "raw", metric)
        mu, se = w.mean(), w.std() / np.sqrt(len(w))
        x = [p*100 for p in PCTS]
        ax.plot(x, mu, "-o", ms=4, color=col[m], label={"RF": "Random Forest", "BRT": "Boosted Trees (XGBoost)"}[m])
        ax.fill_between(x, mu - 1.96*se, mu + 1.96*se, color=col[m], alpha=0.2)
    ax.set_xlabel("Training data (%)"); ax.set_ylabel(f"{metric} (test set)"); ax.grid(alpha=.3)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mae_rmse.png"), dpi=300)

# 7) Figura: matrices de p (Holm) por % de entrenamiento, BRT y RF, MAE
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, m in zip(axes, ["RF", "BRT"]):
    M = pmats[(m, "MAE")]
    im = ax.imshow(M.values < 0.05, cmap="Blues", vmin=0, vmax=1.6)
    for i in range(len(M)):
        for j in range(len(M)):
            v = M.values[i, j]
            ax.text(j, i, "" if np.isnan(v) else (f"{v:.0e}" if v < 1e-3 else f"{v:.3f}"), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(M)), M.columns); ax.set_yticks(range(len(M)), M.index)
    ax.set_title(f"{'Random Forest' if m=='RF' else 'Boosted Trees'} – MAE", fontsize=9)
    ax.set_xlabel("Training data (%)"); ax.set_ylabel("Training data (%)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_pmatrix_mae.png"), dpi=300)
print("\n".join(md))
