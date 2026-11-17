"""Passos 2 e 3 — a rede e o treino (sem regularização além da parada antecipada).

Gera a Figura 1 (curvas de perda de treino e validação por época).
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from comum import CORES, prepara, salva, secao
from modelo import MLP, conta_parametros, descreve
from treino import PADRAO, treina

d = prepara()
D = d.Z_tr.shape[1]

secao("2 — A REDE")
rede = MLP(D, PADRAO["larguras"])
print("arquitetura:", descreve(PADRAO["larguras"], D), "| ReLU nas ocultas | saída = 1 logit")
print(rede)
for nome, p in rede.named_parameters():
    print(f"  {nome:20} {str(tuple(p.shape)):10} {p.numel():6}")
n_par = conta_parametros(rede)
print(f"parâmetros: {n_par} | linhas de sub-treino: {len(d.y_tr)} | "
      f"parâmetros/linha = {n_par / len(d.y_tr):.3f}")
print("perda: BCEWithLogitsLoss(pos_weight = %.4f)" % d.pos_weight)

secao("3A — VARREDURA DE LEARNING RATE (Adam(W), sem regularização, seed 42)")
for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
    t0 = time.time()
    m, h, ep = treina(d, {"lr": lr}, seed=42)
    print(f"lr {lr:<7g} | épocas rodadas {len(h['val']):3} | melhor época {ep:3} | "
          f"perda val {h['val'][ep - 1]:.4f} | PR-AUC val {h['val_prauc'][ep - 1]:.4f} | "
          f"{time.time() - t0:.1f}s")

secao("3B — REDE BASE: lr 1e-3, batch 256, 100 épocas SEM parar (para ver o overfitting)")
t0 = time.time()
_, h, ep = treina(d, {"max_epocas": 100}, seed=42, parada=False)
dur = time.time() - t0
tr, va = np.array(h["treino"]), np.array(h["val"])
# onde a paciência de 15 teria parado: 15 épocas seguidas sem bater o melhor até então
melhor, ep_melhor, parada = np.inf, 0, None
for e, v in enumerate(va, start=1):
    if v < melhor:
        melhor, ep_melhor = v, e
    elif e - ep_melhor == PADRAO["paciencia"]:
        parada = e
        break
print(f"tempo: {dur:.1f}s ({dur / 100:.2f}s por época, {int(np.ceil(len(d.y_tr) / 256))} passos)")
print(f"melhor época (menor perda de validação): {ep} | a parada antecipada pararia na época {parada}")
for e in [1, 5, ep, parada, 50, 100]:
    print(f"  época {e:3} | perda treino {tr[e - 1]:.4f} | perda val {va[e - 1]:.4f} | "
          f"gap {va[e - 1] - tr[e - 1]:+.4f} | PR-AUC val {h['val_prauc'][e - 1]:.4f}")

secao("3C — A MESMA REDE COM PARADA ANTECIPADA (paciência 15, restaura os melhores pesos)")
m, h2, ep2 = treina(d, seed=42)
print(f"rodou {len(h2['val'])} épocas | restaurou a época {ep2} | "
      f"perda val {h2['val'][ep2 - 1]:.4f} | PR-AUC val {h2['val_prauc'][ep2 - 1]:.4f}")

# ---------------------------------------------------------------- Figura 1
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(1, 101)
a.plot(x, tr, color=CORES[3], label="treino (sub-treino)")
a.plot(x, va, color=CORES[0], label="validação")
a.axvline(ep, color=CORES[2], ls="--", label=f"melhor época ({ep})")
if parada:
    a.axvline(parada, color=CORES[1], ls=":", label=f"parada com paciência 15 ({parada})")
a.set_xlabel("época")
a.set_ylabel("BCE com pos_weight")
a.set_title("(a) perda por época")
a.legend()
b.plot(x, h["val_prauc"], color=CORES[0], label="PR-AUC na validação")
b.axvline(ep, color=CORES[2], ls="--", label=f"melhor época ({ep})")
b.set_xlabel("época")
b.set_ylabel("PR-AUC (>50K)")
b.set_title("(b) PR-AUC de validação por época")
b.legend()
fig.suptitle("Figura 1 — Rede base 50→64→32→1 sem regularização: treino × validação (seed 42)")
salva(fig, "fig01-curvas.png")
