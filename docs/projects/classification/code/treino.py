"""O loop de treino do passo 3: mini-batch + AdamW + parada antecipada pela validação.

Segue o esqueleto do enunciado (best / wait / patience / recarregar os melhores pesos),
com os pesos guardados em memória (copy.deepcopy do state_dict) em vez de um arquivo.
"""
import copy

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from modelo import MLP

PADRAO = dict(
    larguras=(64, 32),   # passo 2
    lr=1e-3,             # passo 3: padrão de Adam para MLP tabular (página Optimization)
    wd=0.0,              # weight decay (AdamW, desacoplado); 0 = sem regularização
    dropout=0.0,
    batchnorm=False,
    batch=256,           # ~82 passos por época em 20823 linhas
    max_epocas=200,      # teto folgado: a parada antecipada termina bem antes
    paciencia=15,        # épocas sem melhorar a perda de validação antes de parar
)

torch.set_num_threads(4)


def otimizador(modelo, lr, wd):
    """AdamW com weight decay só nas MATRIZES de pesos (vieses e BatchNorm ficam de fora)."""
    decai = [p for p in modelo.parameters() if p.ndim >= 2]
    nao_decai = [p for p in modelo.parameters() if p.ndim < 2]
    return torch.optim.AdamW([
        {"params": decai, "weight_decay": wd},
        {"params": nao_decai, "weight_decay": 0.0},
    ], lr=lr)


@torch.no_grad()
def probabilidades(modelo, Z):
    modelo.eval()
    return torch.sigmoid(modelo(torch.from_numpy(Z))).numpy()


def treina(d, cfg=None, seed=42, parada=True):
    """Treina uma MLP em d.Z_tr e para pela perda de validação, restaurando os melhores pesos.

    Devolve (modelo com os melhores pesos, histórico por época, melhor época).
    """
    cfg = {**PADRAO, **(cfg or {})}
    torch.manual_seed(seed)
    np.random.seed(seed)
    gerador = torch.Generator().manual_seed(seed)

    X_tr, y_tr = torch.from_numpy(d.Z_tr), torch.from_numpy(d.y_tr.astype("float32"))
    X_val, y_val = torch.from_numpy(d.Z_val), torch.from_numpy(d.y_val.astype("float32"))

    modelo = MLP(X_tr.shape[1], cfg["larguras"], cfg["dropout"], cfg["batchnorm"])
    perda = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(d.pos_weight, dtype=torch.float32))
    opt = otimizador(modelo, cfg["lr"], cfg["wd"])

    hist = {"treino": [], "val": [], "val_prauc": []}
    melhor, espera, melhor_ep, melhores_pesos = float("inf"), 0, 0, None
    n = len(X_tr)
    for epoca in range(1, cfg["max_epocas"] + 1):
        modelo.train()                                   # liga dropout / BatchNorm de treino
        ordem = torch.randperm(n, generator=gerador)
        for i in range(0, n, cfg["batch"]):
            idx = ordem[i:i + cfg["batch"]]
            opt.zero_grad()
            perda(modelo(X_tr[idx]), y_tr[idx]).backward()
            opt.step()

        modelo.eval()                                    # desliga dropout; BN usa médias globais
        with torch.no_grad():
            p_tr = perda(modelo(X_tr), y_tr).item()      # treino medido em modo eval, comparável
            u_val = modelo(X_val)
            p_val = perda(u_val, y_val).item()
        hist["treino"].append(p_tr)
        hist["val"].append(p_val)
        hist["val_prauc"].append(average_precision_score(d.y_val, torch.sigmoid(u_val).numpy()))

        if p_val < melhor:
            melhor, espera, melhor_ep = p_val, 0, epoca
            melhores_pesos = copy.deepcopy(modelo.state_dict())   # cópia, não referência
        else:
            espera += 1
            if parada and espera == cfg["paciencia"]:
                break

    modelo.load_state_dict(melhores_pesos)
    return modelo, hist, melhor_ep
