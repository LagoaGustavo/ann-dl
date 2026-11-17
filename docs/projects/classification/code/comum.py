"""Módulo comum da entrega 2 (classificação) — importável, não executa nada pesado no import.

Reaproveita o pipeline do EDA (docs/projects/eda/code/pipeline.py) SEM copiá-lo: mesmo
arquivo de dados, mesma limpeza, mesmo split treino/teste (80/20 estratificado, seed 42).

    from comum import prepara
    d = prepara()            # sub-treino + validação (o teste fica de fora)
    d = prepara(teste=True)  # só no passo 5, depois de todas as escolhas

A validação sai de DENTRO do treino do EDA (20% estratificado, mesma seed) e o
pré-processador é ajustado SÓ no sub-treino — a validação e o teste são apenas
transformados, então nenhuma estatística (mediana, média, desvio, categorias raras) vem
deles.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(1, str(BASE.parents[1] / "eda" / "code"))  # pipeline.py do EDA
from pipeline import (CATEGORICAL, NUM_PLAIN, NUM_SKEWED, SEED,  # noqa: E402
                      build_preprocessor, load_clean, split)

# o console do Windows (cp1252) não imprime "≤", "—" e "×"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)

# mesma identidade visual do EDA (eda.py:28-34)
CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A", "#7A6C8F"]  # âmbar, telha, oliva, aço, lilás
COR_CLASSE = {0: CORES[3], 1: CORES[0]}
ROTULO = {0: "≤50K", 1: ">50K"}
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})

COLUNAS_BRUTAS = NUM_PLAIN + NUM_SKEWED + CATEGORICAL   # as 12 features depois da limpeza

# ------------------------------------------------------------------ PASSO 0 (antes de treinar)
# Métrica principal: PR-AUC (average precision) da classe >50K. O alvo tem 24.09% de
# positivos (EDA 1C): "sempre ≤50K" já acerta 75.91%, então acurácia não discrimina; a
# PR-AUC não depende de limiar e tem como piso a prevalência.
METRICA = "PR-AUC (average precision) da classe >50K"
# Critério de sucesso: PR-AUC média (seeds) da rede no TESTE >= PR-AUC do melhor baseline
# obrigatório (trivial ou linear, escolhido na validação) + MARGEM, e a diferença maior que
# o desvio padrão da rede entre seeds.
MARGEM = 0.01
# Custo dos erros (para o limiar): cenário de triagem de público de alta renda para uma
# oferta financeira — deixar passar uma pessoa >50K (FN) é uma venda perdida; abordar uma
# pessoa ≤50K (FP) é um contato desperdiçado, mais barato. Assumimos C_FN = 3 × C_FP.
C_FN, C_FP = 3.0, 1.0
LIMIARES = np.round(np.arange(0.01, 1.00, 0.01), 2)

# Seeds: busca (passo 4) com 3 seeds; avaliação final (passo 5) com 5.
SEEDS_BUSCA = [42, 43, 44]
SEEDS_FINAIS = [42, 43, 44, 45, 46]

# Configuração congelada no fim do passo 4 (escolhida SÓ pela validação). O passo 5 usa
# exatamente esta. Preenchida depois de rodar s4_regularizacao.py.
CONFIG_ESCOLHIDA = None


def secao(txt):
    print("\n" + "=" * 80 + f"\n{txt}\n" + "=" * 80, flush=True)


def salva(fig, nome):
    fig.savefig(FIG / nome, bbox_inches="tight")
    plt.close(fig)


def preprocessador(sem=()):
    """O ColumnTransformer do EDA, sem alterar nenhum passo; `sem` tira colunas (ablação)."""
    prep = build_preprocessor()
    if sem:
        prep.transformers = [
            (nome, t, [c for c in cols if c not in sem])
            for nome, t, cols in prep.transformers
            if [c for c in cols if c not in sem]
        ]
    return prep


def prepara(teste=False, sem=()):
    """Split do EDA + validação estratificada dentro do treino; prep ajustado só no sub-treino."""
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)                    # o MESMO split do EDA
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=SEED)

    prep = preprocessador(sem)
    d = SimpleNamespace(
        X_tr=X_tr, X_val=X_val,
        y_tr=y_tr.to_numpy(), y_val=y_val.to_numpy(),
        Z_tr=prep.fit_transform(X_tr).astype("float32"),             # fit SÓ no sub-treino
        Z_val=prep.transform(X_val).astype("float32"),
        prep=prep, n_train_eda=len(X_train),
    )
    d.nomes = list(prep.get_feature_names_out())
    n_pos = int(d.y_tr.sum())
    d.pos_weight = (len(d.y_tr) - n_pos) / n_pos                   # N_neg / N_pos do sub-treino
    if teste:                                                       # só no passo 5
        d.X_te, d.y_te = X_test, y_test.to_numpy()
        d.Z_te = prep.transform(X_test).astype("float32")
    return d


def custo(y, p, t):
    """Custo médio por pessoa no limiar t: (C_FN·FN + C_FP·FP) / n."""
    pred = p >= t
    fn = np.sum((y == 1) & ~pred)
    fp = np.sum((y == 0) & pred)
    return (C_FN * fn + C_FP * fp) / len(y)


def limiar_por_custo(y, p):
    """Varre os limiares 0.01…0.99 e devolve (melhor limiar, custos de cada limiar)."""
    custos = np.array([custo(y, p, t) for t in LIMIARES])
    return float(LIMIARES[int(np.argmin(custos))]), custos


def avalia(y, p, t):
    """Todas as métricas pedidas para classificação, num limiar t."""
    pred = (p >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
        "limiar": t, "acuracia": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "prec_0": prec[0], "rec_0": rec[0], "f1_0": f1[0],
        "prec_1": prec[1], "rec_1": rec[1], "f1_1": f1[1],
        "custo": custo(y, p, t), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def linha(nome, m, chaves=("pr_auc", "roc_auc", "limiar", "acuracia", "f1_macro",
                           "prec_1", "rec_1", "f1_1", "custo")):
    return f"{nome:34} | " + " | ".join(f"{k} {m[k]:.4f}" for k in chaves)
