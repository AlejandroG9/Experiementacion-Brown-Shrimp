"""
Experimento corregido para la revisión IJCOPI (Revisor A y Revisor B #13).

Diseño:
  - R réplicas independientes; la réplica r usa la semilla 1000 + r para la partición
    train/test y para los modelos. La MISMA partición se usa para RF y BRT y para las
    versiones normalizada / no normalizada  ->  observaciones pareadas reales.
  - Proporciones de entrenamiento: 20 % ... 80 %.
  - Ajuste de hiperparámetros: RandomizedSearchCV con el MISMO presupuesto (N_ITER
    combinaciones, CV de K pliegues) para ambos algoritmos, usando SOLO el conjunto
    de entrenamiento. El conjunto de prueba no participa en la selección del modelo.
  - Normalización Min-Max ajustada solo con el entrenamiento y aplicada a las variables
    predictoras; la variable respuesta se conserva en sus unidades originales, de modo
    que MAE y RMSE son comparables entre ambas versiones.
  - Se guarda, por corrida: MAE, RMSE, R2, mejores hiperparámetros, tiempos.

Uso:
  python experimento.py --reps 30 --out resultados.csv
  python experimento.py --reps 1 --pcts 0.5 --out piloto.csv     (prueba rápida)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "Estadisticas V5.csv")
TARGET = "CapturaTotal"
DROP = ["Unnamed: 0", "Fecha.Completa", "CafeLineaSC", "CafeLineaCC",
        "CafePacotillaSC", "CafePacotillaCC", "Mes", "Anno"]

# Espacios de búsqueda de tamaño equivalente (36 combinaciones cada uno)
RF_SPACE = {
    "n_estimators": [300, 600],
    "max_features": [0.33, 0.5, 1.0],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5],
}
XGB_SPACE = {
    "n_estimators": [300, 600],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 6, 10],
    "subsample": [0.8, 1.0],
}


def load():
    d = pd.read_csv(DATA).drop(columns=DROP)
    return d.drop(columns=[TARGET]), d[TARGET].to_numpy()


def search(est, space, Xtr, ytr, seed, n_iter, k, n_jobs):
    s = RandomizedSearchCV(est, space, n_iter=n_iter, cv=KFold(k, shuffle=True, random_state=seed),
                           scoring="neg_mean_squared_error", random_state=seed, n_jobs=n_jobs, refit=True)
    s.fit(Xtr, ytr)
    return s


def metrics(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--pcts", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out", type=str, default="resultados.csv")
    a = ap.parse_args()
    out = os.path.join(HERE, a.out)
    pcts = [float(x) for x in a.pcts.split(",")]

    X, y = load()
    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.rep, prev.pct.round(2), prev.scaling, prev.model))
    print(f"Datos: {X.shape}, variables: {list(X.columns)}; ya hechas: {len(done)}", flush=True)

    for r in range(1, a.reps + 1):
        seed = 1000 + r
        for p in pcts:
            Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=p, random_state=seed)
            for scaling in ["raw", "minmax"]:
                if scaling == "minmax":
                    sc = MinMaxScaler().fit(Xtr)
                    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
                else:
                    Xtr_s, Xte_s = Xtr.to_numpy(), Xte.to_numpy()
                models = {
                    "RF": lambda: search(RandomForestRegressor(random_state=seed, n_jobs=1), RF_SPACE,
                                         Xtr_s, ytr, seed, a.n_iter, a.k, a.n_jobs),
                    "BRT": lambda: search(XGBRegressor(random_state=seed, n_jobs=1, objective="reg:squarederror",
                                                       tree_method="hist"), XGB_SPACE,
                                          Xtr_s, ytr, seed, a.n_iter, a.k, a.n_jobs),
                }
                rows = []
                for name, fn in models.items():
                    if (r, round(p, 2), scaling, name) in done:
                        continue
                    t0 = time.time()
                    s = fn()
                    pred = s.best_estimator_.predict(Xte_s)
                    rows.append(dict(rep=r, seed=seed, pct=round(p, 2), scaling=scaling, model=name,
                                     n_train=len(ytr), n_test=len(yte), **metrics(yte, pred),
                                     cv_RMSE=float(np.sqrt(-s.best_score_)),
                                     best_params=json.dumps(s.best_params_), seconds=round(time.time() - t0, 1)))
                # Línea base (predice la media del entrenamiento) para contexto
                if (r, round(p, 2), scaling, "Mean") not in done:
                    dm = DummyRegressor().fit(Xtr_s, ytr)
                    rows.append(dict(rep=r, seed=seed, pct=round(p, 2), scaling=scaling, model="Mean",
                                     n_train=len(ytr), n_test=len(yte), **metrics(yte, dm.predict(Xte_s)),
                                     cv_RMSE=np.nan, best_params="{}", seconds=0))
                if rows:
                    df = pd.DataFrame(rows)
                    df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
                    for _, w in df.iterrows():
                        print(f"rep {r:2d} p={p:.1f} {scaling:6s} {w.model:4s} MAE={w.MAE:7.3f} "
                              f"RMSE={w.RMSE:7.3f} R2={w.R2:6.3f} ({w.seconds}s)", flush=True)


if __name__ == "__main__":
    main()
