"""Passos 5 e 6 — avaliação ÚNICA no teste e análise de erros.

Só existe depois do commit que congelou CONFIG_ESCOLHIDA (passo 4): nada aqui volta a
mexer em configuração ou limiar. O limiar de cada modelo é escolhido NA VALIDAÇÃO.
Gera as Figuras 3 a 7.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve, roc_curve)

from comum import (C_FN, C_FP, COLUNAS_BRUTAS, CONFIG_ESCOLHIDA, CORES, LIMIARES, MARGEM,
                   SEED, SEEDS_BUSCA, SEEDS_FINAIS, avalia, limiar_por_custo, linha, prepara,
                   salva, secao)
from modelo import MLP, conta_parametros, descreve
from treino import PADRAO, probabilidades, treina

assert CONFIG_ESCOLHIDA is not None, "congele a configuração no passo 4 antes do teste"
cfg = CONFIG_ESCOLHIDA
d = prepara(teste=True)
D = d.Z_tr.shape[1]

secao("5 — CONFIGURAÇÃO ESCOLHIDA (congelada no passo 4, pela validação)")
cfg_total = {**PADRAO, **cfg}
print("cfg:", cfg_total)
print("arquitetura:", descreve(cfg_total["larguras"], D), "| parâmetros:",
      conta_parametros(MLP(D, cfg_total["larguras"], cfg_total["dropout"])))
print(f"linhas: sub-treino {len(d.y_tr)} | validação {len(d.y_val)} | teste {len(d.y_te)} "
      f"(>50K no teste: {int(d.y_te.sum())}, {d.y_te.mean() * 100:.2f}%)")

secao("5 — RETREINO COM 5 SEEDS (parada e limiar decididos na validação)")
redes = []
for s in SEEDS_FINAIS:
    m, h, ep = treina(d, cfg, seed=s)
    p_val = probabilidades(m, d.Z_val)
    t, custos = limiar_por_custo(d.y_val, p_val)
    redes.append(dict(seed=s, modelo=m, ep=ep, epocas=len(h["val"]), t=t, custos=custos,
                      p_val=p_val, p_te=probabilidades(m, d.Z_te)))
    print(f"seed {s} | rodou {len(h['val'])} épocas, restaurou a {ep} | limiar (val) {t:.2f} | "
          f"PR-AUC val {average_precision_score(d.y_val, p_val):.4f}")
print("(as seeds 42-44 repetem exatamente as corridas da busca; 45 e 46 são novas)")

baselines = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "Logística": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    "HistGradientBoosting (opcional)": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=SEED),
}
base = {}
for nome, mod in baselines.items():
    mod.fit(d.Z_tr, d.y_tr)
    p_val = mod.predict_proba(d.Z_val)[:, 1]
    t, custos = limiar_por_custo(d.y_val, p_val)
    base[nome] = dict(t=t, custos=custos, p_val=p_val, p_te=mod.predict_proba(d.Z_te)[:, 1],
                      pr_val=average_precision_score(d.y_val, p_val))

secao("5 — TESTE (usado UMA vez)")
teste = {n: avalia(d.y_te, b["p_te"], b["t"]) for n, b in base.items()}
for n, m in teste.items():
    print(linha(n, m))
por_seed = pd.DataFrame([{"seed": r["seed"], **avalia(d.y_te, r["p_te"], r["t"])} for r in redes])
for _, r in por_seed.iterrows():
    print(linha(f"MLP seed {int(r.seed)}", r))
chaves = ["pr_auc", "roc_auc", "acuracia", "f1_macro", "prec_0", "rec_0", "f1_0",
          "prec_1", "rec_1", "f1_1", "custo", "limiar"]
media, dp = por_seed[chaves].mean(), por_seed[chaves].std(ddof=1)
print("MLP média ± dp (5 seeds):")
for k in chaves:
    print(f"  {k:9} {media[k]:.4f} ± {dp[k]:.4f}")

r42 = redes[0]
pred42 = (r42["p_te"] >= r42["t"]).astype(int)
print("\nclassification_report — MLP seed 42, limiar", r42["t"])
print(classification_report(d.y_te, pred42, target_names=["≤50K", ">50K"], digits=3))
m42 = avalia(d.y_te, r42["p_te"], r42["t"])
print(f"matriz de confusão (seed 42): TN {m42['tn']} | FP {m42['fp']} | FN {m42['fn']} | TP {m42['tp']}")

secao("5 — CRITÉRIO DO PASSO 0")
melhor_obrig = max(["Dummy (most_frequent)", "Logística"], key=lambda n: base[n]["pr_val"])
ref = teste[melhor_obrig]["pr_auc"]
dif = media["pr_auc"] - ref
print(f"melhor baseline obrigatório (escolhido pela PR-AUC de VALIDAÇÃO): {melhor_obrig}")
print(f"PR-AUC teste: MLP {media['pr_auc']:.4f} ± {dp['pr_auc']:.4f} | {melhor_obrig} {ref:.4f} | "
      f"diferença {dif:+.4f} (exigido >= {MARGEM} e > dp {dp['pr_auc']:.4f})")
print("CRITÉRIO ATINGIDO" if (dif >= MARGEM and dif > dp["pr_auc"]) else "CRITÉRIO NÃO ATINGIDO")
hgb = teste["HistGradientBoosting (opcional)"]["pr_auc"]
print(f"referência opcional HistGradientBoosting: PR-AUC teste {hgb:.4f} "
      f"(MLP − HGB = {media['pr_auc'] - hgb:+.4f})")

# ---------------------------------------------------------------- Figura 3: custo × limiar
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(LIMIARES, r42["custos"], color=CORES[0], label="MLP (seed 42)")
ax.plot(LIMIARES, base["Logística"]["custos"], color=CORES[3], label="Logística")
ax.axhline(C_FN * d.y_val.mean(), color="gray", ls=":", label="Dummy (sempre ≤50K)")
ax.axvline(r42["t"], color=CORES[0], ls="--", label=f"limiar MLP = {r42['t']:.2f}")
ax.axvline(base["Logística"]["t"], color=CORES[3], ls="--",
           label=f"limiar logística = {base['Logística']['t']:.2f}")
ax.axvline(0.5, color=CORES[1], ls="-.", lw=1, label="0.5 (padrão)")
ax.set_xlabel("limiar sobre a probabilidade prevista")
ax.set_ylabel(f"custo médio por pessoa ({C_FN:g}·FN + {C_FP:g}·FP)/n")
ax.set_title("Figura 3 — Custo × limiar na validação (C_FN = 3 × C_FP)")
ax.legend(fontsize=8)
salva(fig, "fig03-limiar.png")

# ---------------------------------------------------------------- Figura 4: matriz de confusão
cm = np.array([[m42["tn"], m42["fp"]], [m42["fn"], m42["tp"]]])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum() * 100:.1f}% da linha)",
                ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["prevê ≤50K", "prevê >50K"])
ax.set_yticks([0, 1], ["real ≤50K", "real >50K"])
ax.set_xlabel("classe prevista")
ax.set_ylabel("classe real")
ax.grid(False)
fig.colorbar(im, ax=ax, label="nº de pessoas")
ax.set_title(f"Figura 4 — Matriz de confusão no teste\n(MLP seed 42, limiar {r42['t']:.2f})")
salva(fig, "fig04-confusao.png")

# ---------------------------------------------------------------- Figura 5: PR e ROC
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
curvas = [("MLP (seed 42)", r42["p_te"], CORES[0]), ("Logística", base["Logística"]["p_te"], CORES[3]),
          ("HistGradientBoosting", base["HistGradientBoosting (opcional)"]["p_te"], CORES[2])]
for nome, p, cor in curvas:
    pr, rc, _ = precision_recall_curve(d.y_te, p)
    a.plot(rc, pr, color=cor, label=f"{nome} (PR-AUC {average_precision_score(d.y_te, p):.3f})")
    fpr, tpr, _ = roc_curve(d.y_te, p)
    b.plot(fpr, tpr, color=cor, label=f"{nome} (ROC-AUC {avalia(d.y_te, p, 0.5)['roc_auc']:.3f})")
a.axhline(d.y_te.mean(), color="gray", ls=":", label=f"Dummy = prevalência ({d.y_te.mean():.3f})")
a.set_xlabel("recall (>50K)")
a.set_ylabel("precisão (>50K)")
a.set_title("(a) precisão × recall")
a.legend(fontsize=8)
b.plot([0, 1], [0, 1], color="gray", ls=":", label="Dummy (ROC-AUC 0.5)")
b.set_xlabel("taxa de falsos positivos")
b.set_ylabel("taxa de verdadeiros positivos")
b.set_title("(b) ROC")
b.legend(fontsize=8)
fig.suptitle("Figura 5 — Curvas no teste: MLP × baselines")
salva(fig, "fig05-pr-roc.png")

# ================================================================ PASSO 6
secao("6 — ERRO POR SUBGRUPO (teste; média das 5 seeds, cada uma no seu limiar)")
X = d.X_te.reset_index(drop=True).copy()
grupos = {
    "sex": X["sex"],
    "race": X["race"].where(X["race"].isin(["White", "Black", "Asian-Pac-Islander"]), "outras (raras)"),
    "relationship": X["relationship"],
    "marital-status": X["marital-status"],
    "workclass": X["workclass"].fillna("Unknown (faltante)"),
    "native-country": X["native-country"].fillna("Unknown").where(
        X["native-country"].fillna("Unknown").isin(["United-States", "Mexico", "Unknown"]), "outros (raros)"),
    "capital": np.select([X["capital-gain"] > 0, X["capital-loss"] > 0],
                         ["ganho > 0", "perda > 0"], "sem ganho nem perda"),
    "capital-gain = 99999": np.where(X["capital-gain"] == 99999, "sim", "não"),
    "idade": pd.cut(X["age"], [0, 29, 50, 200], labels=["< 30", "30–50", "> 50"]).astype(str),
    "education-num": pd.cut(X["education-num"], [0, 9, 12, 16],
                            labels=["≤ 9 (até HS-grad)", "10–12", "≥ 13 (Bachelors+)"]).astype(str),
    "classe real": np.where(d.y_te == 1, ">50K", "≤50K"),
}
linhas_sg = []
for var, g in grupos.items():
    g = pd.Series(np.asarray(g))
    for nivel in g.unique():
        msk = (g == nivel).to_numpy()
        y = d.y_te[msk]
        err, fnr, fpr = [], [], []
        for r in redes:
            pr_ = r["p_te"][msk] >= r["t"]
            err.append(np.mean(pr_ != y))
            fnr.append(np.mean(~pr_[y == 1]) if (y == 1).any() else np.nan)
            fpr.append(np.mean(pr_[y == 0]) if (y == 0).any() else np.nan)
        linhas_sg.append({"variável": var, "grupo": nivel, "n": int(msk.sum()),
                          "prev_>50K": y.mean(), "erro": np.mean(err), "erro_dp": np.std(err, ddof=1),
                          "FNR": np.nanmean(fnr) if not np.all(np.isnan(fnr)) else np.nan,
                          "FPR": np.nanmean(fpr) if not np.all(np.isnan(fpr)) else np.nan})
sg = pd.DataFrame(linhas_sg)
print(sg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nerro global da MLP no teste: {1 - media['acuracia']:.4f}")
sg_n = sg[(sg.n >= 100) & (sg["variável"] != "classe real")].sort_values("erro", ascending=False)
pior = sg_n.iloc[0]
print(f"subgrupo (n ≥ 100) com MAIOR erro: {pior['variável']} = {pior.grupo} | erro {pior.erro:.4f} "
      f"| n {pior.n} | prevalência {pior['prev_>50K']:.4f} | FNR {pior.FNR:.4f} | FPR {pior.FPR:.4f}")
print("top 5 (n ≥ 100):")
print(sg_n.head(5).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

secao("6 — PIORES EXEMPLOS (MLP seed 42)")
Xs = X.assign(income=np.where(d.y_te == 1, ">50K", "≤50K"), p=np.round(r42["p_te"], 4))
fp_ = Xs[(d.y_te == 0) & (pred42 == 1)].sort_values("p", ascending=False).head(5)
fn_ = Xs[(d.y_te == 1) & (pred42 == 0)].sort_values("p").head(5)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print("5 falsos positivos mais confiantes (real ≤50K, maior p):")
    print(fp_.to_string(index=False))
    print("\n5 falsos negativos mais confiantes (real >50K, menor p):")
    print(fn_.to_string(index=False))

secao("6 — PERMUTATION IMPORTANCE (validação, MLP seed 42, 5 repetições por coluna bruta)")
rng = np.random.default_rng(SEED)
base_pr = average_precision_score(d.y_val, r42["p_val"])
imp = []
for col in COLUNAS_BRUTAS:
    quedas = []
    for _ in range(5):
        Xp = d.X_val.copy()
        Xp[col] = rng.permutation(Xp[col].to_numpy())
        p = probabilidades(r42["modelo"], d.prep.transform(Xp).astype("float32"))
        quedas.append(base_pr - average_precision_score(d.y_val, p))
    imp.append({"coluna": col, "queda_prauc": np.mean(quedas), "dp": np.std(quedas, ddof=1)})
imp = pd.DataFrame(imp).sort_values("queda_prauc", ascending=False).reset_index(drop=True)
print(f"PR-AUC de validação sem permutar: {base_pr:.4f}")
print(imp.to_string(float_format=lambda v: f"{v:.4f}"))
EDA = ["relationship", "marital-status", "capital-gain", "education-num", "age", "occupation",
       "hours-per-week", "sex", "workclass", "capital-loss", "race", "native-country"]
print("ordem sugerida pelo EDA (amplitude da taxa de >50K e cargas da PC1):", EDA)
rank_eda = {c: i for i, c in enumerate(EDA)}
rk = pd.Series([rank_eda[c] for c in imp.coluna]).corr(pd.Series(range(len(imp))), method="spearman")
print(f"correlação de Spearman entre os dois rankings: {rk:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.barh(imp.coluna, imp.queda_prauc, xerr=imp.dp, color=CORES[0], label="média ± dp (5 permutações)")
ax.invert_yaxis()
ax.set_xlabel("queda da PR-AUC de validação ao embaralhar a coluna")
ax.set_ylabel("feature (coluna bruta)")
ax.set_title("Figura 7 — Permutation importance da MLP (validação, seed 42)")
ax.legend(loc="lower right")
salva(fig, "fig07-permutacao.png")

secao("6 — ABLAÇÕES PROMETIDAS NO EDA (validação, config escolhida, 3 seeds)")
abl = {"todas as features": (), "sem capital-gain e capital-loss": ("capital-gain", "capital-loss"),
       "sem sex, race e relationship": ("sex", "race", "relationship")}
for nome, sem in abl.items():
    da = prepara(sem=sem)
    prs = [average_precision_score(da.y_val, probabilidades(treina(da, cfg, seed=s)[0], da.Z_val))
           for s in SEEDS_BUSCA]
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(da.Z_tr, da.y_tr)
    print(f"{nome:34} | {da.Z_tr.shape[1]:2} colunas | MLP PR-AUC val {np.mean(prs):.4f} ± "
          f"{np.std(prs, ddof=1):.4f} | logística {average_precision_score(da.y_val, lr.predict_proba(da.Z_val)[:, 1]):.4f}")

sem_cap = ((X["capital-gain"] == 0) & (X["capital-loss"] == 0)).to_numpy()
prs = [average_precision_score(d.y_te[sem_cap], r["p_te"][sem_cap]) for r in redes]
print(f"\nteste, subconjunto sem ganho nem perda de capital (n {sem_cap.sum()}, "
      f"prevalência {d.y_te[sem_cap].mean():.4f}): MLP PR-AUC {np.mean(prs):.4f} ± {np.std(prs, ddof=1):.4f}"
      f" | logística {average_precision_score(d.y_te[sem_cap], base['Logística']['p_te'][sem_cap]):.4f}")

# ---------------------------------------------------------------- Figura 6: subgrupos
mostra = sg[sg["variável"].isin(["sex", "race", "relationship", "capital", "idade"])].copy()
mostra["rótulo"] = mostra["variável"] + " = " + mostra.grupo.astype(str) + "  (n=" + mostra.n.astype(str) + ")"
mostra = mostra.iloc[::-1]
y = np.arange(len(mostra))
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(y - 0.27, mostra.erro, height=0.27, color=CORES[3], label="taxa de erro")
ax.barh(y, mostra.FNR, height=0.27, color=CORES[0], label="FNR (>50K perdidos)")
ax.barh(y + 0.27, mostra.FPR, height=0.27, color=CORES[1], label="FPR (≤50K marcados como >50K)")
ax.axvline(1 - media["acuracia"], color="gray", ls=":", label=f"erro global ({1 - media['acuracia']:.3f})")
ax.set_yticks(y, mostra["rótulo"])
ax.set_xlabel("taxa (média das 5 seeds, teste)")
ax.set_ylabel("subgrupo")
ax.set_title("Figura 6 — Erro, FNR e FPR da MLP por subgrupo (teste)")
ax.legend(loc="lower right", fontsize=8)
salva(fig, "fig06-subgrupos.png")
