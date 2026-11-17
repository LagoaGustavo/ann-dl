"""Passo 1 — baselines no MESMO pipeline, avaliados na VALIDAÇÃO.

Trivial: DummyClassifier(most_frequent). Linear: regressão logística. Opcional:
HistGradientBoosting (árvores), a referência forte em dado tabular.
O limiar de cada modelo é escolhido na validação pelo custo (C_FN = 3 × C_FP).
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from comum import SEED, avalia, limiar_por_custo, linha, prepara, secao

d = prepara()
modelos = {
    "Dummy (most_frequent)": DummyClassifier(strategy="most_frequent"),
    "Logística (class_weight=balanced)": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED),
    "HistGradientBoosting (opcional)": HistGradientBoostingClassifier(
        class_weight="balanced", random_state=SEED),
}

secao("1 — BASELINES NA VALIDAÇÃO (limiar escolhido pelo custo na validação)")
for nome, m in modelos.items():
    m.fit(d.Z_tr, d.y_tr)
    p = m.predict_proba(d.Z_val)[:, 1]
    t, _ = limiar_por_custo(d.y_val, p)
    print(linha(nome, avalia(d.y_val, p, t)))
    if nome.startswith("Dummy"):
        a = avalia(d.y_val, p, 0.5)
        print(f"{'':34} | acurácia do Dummy = {a['acuracia']:.4f} (= fração de ≤50K na validação)")
    if nome.startswith("Logística"):
        print(f"{'':34} | acurácia no limiar 0.5 = {avalia(d.y_val, p, 0.5)['acuracia']:.4f}")
