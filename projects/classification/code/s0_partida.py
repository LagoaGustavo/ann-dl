"""Passo 0 — ponto de partida: pipeline do EDA importado, validação, métrica e critério.

Não treina nada. Só confere que o split é o do EDA, cria a validação dentro do treino,
ajusta o pré-processador no sub-treino e imprime os números citados no relatório.
"""
import numpy as np

from comum import (C_FN, C_FP, MARGEM, METRICA, SEED, prepara, secao)

secao("0 — SPLIT DO EDA + VALIDAÇÃO")
d = prepara()
print(f"treino do EDA: {d.n_train_eda} linhas (seed {SEED}, 80/20 estratificado)")
for nome, y in [("sub-treino", d.y_tr), ("validação", d.y_val)]:
    print(f"  {nome:11} {len(y):6} linhas | >50K {int(y.sum()):5} ({y.mean() * 100:.2f}%)")
print(f"pos_weight = N_neg/N_pos do sub-treino = {len(d.y_tr) - d.y_tr.sum()}/{d.y_tr.sum()}"
      f" = {d.pos_weight:.4f}")

secao("0 — PIPELINE AJUSTADO SÓ NO SUB-TREINO")
print("shape sub-treino:", d.Z_tr.shape, "| shape validação:", d.Z_val.shape)
print("NaN restantes:", int(np.isnan(d.Z_tr).sum()), "/", int(np.isnan(d.Z_val).sum()))
num = d.Z_tr[:, :5]
print("média das 5 numéricas no sub-treino:", np.round(num.mean(axis=0), 4))
print("média das 5 numéricas na validação: ", np.round(d.Z_val[:, :5].mean(axis=0), 4))
print(f"{len(d.nomes)} features:", d.nomes)

secao("0 — MÉTRICA E CRITÉRIO (escritos antes de treinar)")
print("métrica principal:", METRICA)
print(f"critério: PR-AUC média da rede no teste >= melhor baseline obrigatório + {MARGEM}"
      " e diferença > desvio padrão entre seeds")
print(f"custos para o limiar: C_FN = {C_FN}, C_FP = {C_FP}"
      f" -> limiar de Bayes com probabilidades calibradas = C_FP/(C_FP+C_FN) = "
      f"{C_FP / (C_FP + C_FN):.2f}")
