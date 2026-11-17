"""Passo 4 — regularização e busca de hiperparâmetros, TUDO decidido na validação.

A) A mesma rede (64, 32), lr 1e-3, mudando UM fator por vez: nenhuma regularização,
   weight decay (2 valores), dropout, BatchNorm, weight decay + dropout. 3 seeds cada.
B) Busca em grade: largura/profundidade × learning rate × regularização (24 combinações,
   3 delas iguais a variantes da parte A — 27 configurações únicas no total).
O critério de escolha é a PR-AUC média de validação nas 3 seeds (empate → rede menor).
Gera a Figura 2 e o arquivo busca_resultados.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from comum import BASE, CORES, SEEDS_BUSCA, prepara, salva, secao
from modelo import MLP, conta_parametros, descreve
from treino import treina

d = prepara()
D = d.Z_tr.shape[1]


def roda(nome, cfg, parte):
    res = []
    for s in SEEDS_BUSCA:
        _, h, ep = treina(d, cfg, seed=s)
        res.append((h["val_prauc"][ep - 1], h["val"][ep - 1], h["treino"][ep - 1], ep))
    pr, lv, lt, ep = map(np.array, zip(*res))
    larg = cfg.get("larguras", (64, 32))
    linha = {
        "parte": parte, "config": nome, "arquitetura": descreve(larg, D),
        "parametros": conta_parametros(MLP(D, larg, cfg.get("dropout", 0),
                                           cfg.get("batchnorm", False))),
        "lr": cfg.get("lr", 1e-3), "wd": cfg.get("wd", 0.0), "dropout": cfg.get("dropout", 0.0),
        "batchnorm": cfg.get("batchnorm", False),
        "prauc_val_media": pr.mean(), "prauc_val_dp": pr.std(ddof=1),
        "perda_val": lv.mean(), "gap_val_menos_treino": (lv - lt).mean(),
        "melhor_epoca_media": ep.mean(), "cfg": cfg,
    }
    print(f"{parte} | {nome:42} | PR-AUC val {pr.mean():.4f} ± {pr.std(ddof=1):.4f} | "
          f"perda val {lv.mean():.4f} | gap {linha['gap_val_menos_treino']:+.4f} | "
          f"época {ep.mean():5.1f} | {linha['parametros']} parâmetros", flush=True)
    return linha


secao("4A — UM FATOR POR VEZ (rede 64-32, lr 1e-3, 3 seeds)")
VARIANTES = {
    "nenhuma": {},
    "weight decay 0.01": {"wd": 1e-2},
    "weight decay 0.1": {"wd": 1e-1},
    "dropout 0.3": {"dropout": 0.3},
    "BatchNorm": {"batchnorm": True},
    "weight decay 0.1 + dropout 0.3": {"wd": 1e-1, "dropout": 0.3},
}
linhas = [roda(n, c, "A") for n, c in VARIANTES.items()]


def chave(cfg):
    c = {"larguras": (64, 32), "lr": 1e-3, "wd": 0.0, "dropout": 0.0, "batchnorm": False, **cfg}
    return (tuple(c["larguras"]), c["lr"], c["wd"], c["dropout"], c["batchnorm"])


ja_rodadas = {chave(l["cfg"]): l for l in linhas}

secao("4B — BUSCA: largura/profundidade × lr × regularização (3 seeds cada)")
REGS = {"wd 0.1": {"wd": 1e-1}, "dropout 0.3": {"dropout": 0.3},
        "wd 0.1 + dropout 0.3": {"wd": 1e-1, "dropout": 0.3}}
for larg in [(64,), (64, 32), (128, 64), (128, 64, 32)]:
    for lr in [1e-3, 3e-3]:
        for nr, reg in REGS.items():
            cfg = {"larguras": larg, "lr": lr, **reg}
            nome = f"{'-'.join(map(str, larg))} | lr {lr:g} | {nr}"
            if chave(cfg) in ja_rodadas:              # já está na parte A: não conta duas vezes
                ja_rodadas[chave(cfg)]["parte"] = "A+B"
                print(f"B | {nome:42} | = variante da parte A (não repetida)")
                continue
            linhas.append(roda(nome, cfg, "B"))

tab = pd.DataFrame(linhas)
tab = tab.sort_values(["prauc_val_media", "parametros"], ascending=[False, True])
tab.drop(columns="cfg").to_csv(BASE / "busca_resultados.csv", index=False, float_format="%.4f")

secao(f"4 — TABELA ORDENADA PELA PR-AUC DE VALIDAÇÃO ({len(tab)} configurações únicas testadas)")
cols = ["parte", "config", "parametros", "prauc_val_media", "prauc_val_dp", "perda_val",
        "gap_val_menos_treino", "melhor_epoca_media"]
print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
top = tab.iloc[0]
empate = tab[tab.prauc_val_media >= top.prauc_val_media - top.prauc_val_dp]
print(f"\nmelhor: {top.config} | PR-AUC val {top.prauc_val_media:.4f} ± {top.prauc_val_dp:.4f}")
print(f"configs a menos de 1 dp da melhor: {len(empate)}")
escolhida = tab.iloc[0]
print("ESCOLHIDA (maior PR-AUC média de validação):", escolhida.config, "->", escolhida.cfg)

# ---------------------------------------------------------------- Figura 2
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained",
                           gridspec_kw={"width_ratios": [1.15, 1]})
cores = dict(zip(VARIANTES, [CORES[1], CORES[0], CORES[2], CORES[3], CORES[4], "#444444"]))
for nome, cfg in VARIANTES.items():
    _, h, _ = treina(d, {**cfg, "max_epocas": 80}, seed=42, parada=False)
    a.plot(np.arange(1, 81), h["val"], color=cores[nome], label=nome)
a.set_xlabel("época")
a.set_ylabel("perda de validação (BCE com pos_weight)")
a.set_title("(a) perda de validação, 80 épocas sem parar (seed 42)")
a.legend(fontsize=8)
parteA = tab[tab.parte.str.startswith("A")].set_index("config").loc[list(VARIANTES)]
y = np.arange(len(parteA))
b.barh(y, parteA.prauc_val_media, color=[cores[n] for n in parteA.index])
b.errorbar(parteA.prauc_val_media, y, xerr=parteA.prauc_val_dp, fmt="none", ecolor="black",
           capsize=3, label="média ± dp (3 seeds)")
b.set_yticks(y, parteA.index)
b.invert_yaxis()
b.set_xlim(parteA.prauc_val_media.min() - 0.01,
           (parteA.prauc_val_media + parteA.prauc_val_dp).max() + 0.005)
b.set_xlabel("PR-AUC na validação (com parada antecipada)")
b.set_ylabel("regularização (rede 64-32, lr 1e-3)")
b.set_title("(b) PR-AUC de validação por regularização")
b.legend(loc="lower right", fontsize=8)
fig.suptitle("Figura 2 — Regularização: rede 64-32, lr 1e-3, um fator por vez (validação)")
salva(fig, "fig02-regularizacao.png")
