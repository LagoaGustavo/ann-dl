"""A rede do passo 2: MLP em PyTorch que devolve LOGITS (sem sigmoide no fim)."""
import torch
from torch import nn


class MLP(nn.Module):
    """Linear → [BatchNorm] → ReLU → [Dropout], repetido por camada oculta; saída = 1 logit.

    - ReLU nas ocultas: derivada 1 para x > 0, sem a saturação da sigmoide/tanh (derivada
      ≤ 0.25 / ≤ 1) que encolhe o gradiente camada a camada;
    - inicialização de He (Kaiming normal) nas camadas seguidas de ReLU: mantém a variância
      das ativações estável quando metade delas é zerada pela ReLU; Glorot (Xavier) na saída,
      que é linear; vieses começam em zero;
    - a saída é UM logit u: a probabilidade é σ(u), aplicada só na hora de prever, porque a
      perda BCEWithLogitsLoss junta sigmoide + entropia cruzada de forma numericamente estável.
    """

    def __init__(self, d_entrada, larguras=(64, 32), dropout=0.0, batchnorm=False):
        super().__init__()
        camadas, d = [], d_entrada
        for h in larguras:
            camadas.append(nn.Linear(d, h))
            if batchnorm:
                camadas.append(nn.BatchNorm1d(h))
            camadas.append(nn.ReLU())
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            d = h
        self.ocultas = nn.Sequential(*camadas)
        self.saida = nn.Linear(d, 1)
        self._inicializa()

    def _inicializa(self):
        for m in self.ocultas.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.saida.weight)
        nn.init.zeros_(self.saida.bias)

    def forward(self, x):
        return self.saida(self.ocultas(x)).squeeze(1)      # (N,) logits


def conta_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def descreve(larguras, d_entrada):
    """Ex.: '50 → 64 → 32 → 1'."""
    return " → ".join(str(x) for x in (d_entrada, *larguras, 1))
