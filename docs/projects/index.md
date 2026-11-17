# Projeto

!!! abstract "Enunciados"

    [Projects → EDA](https://insper.github.io/ann-dl/2026.2/projects/eda/){:target='_blank'}
    · [Projects → Classification or Regression](https://insper.github.io/ann-dl/2026.2/projects/supervised/){:target='_blank'}

O projeto é **um só**, feito em equipe sobre **o mesmo dataset**, e entregue em três partes
ao longo do semestre, cada uma com data e peso próprios.

## Equipe

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Gustavo Doniani Lagôa Gomes | gustavog6@al.insper.edu.br | [@LagoaGustavo](https://github.com/LagoaGustavo) (antes [@Lagoass](https://github.com/Lagoass)) |
| Deena El Orra | deenaeo@al.insper.edu.br | [@DeenaElOrra](https://github.com/DeenaElOrra) |

Os nomes se repetem no cabeçalho de cada entrega — quem corrige pode abrir uma página
sozinha, sem passar por aqui.

## As três entregas

| # | Entrega | Prazo | Peso no projeto | Página |
|---|---------|-------|-----------------|--------|
| 1 | EDA | 08/out | 20% | [EDA](eda/index.md) |
| 2 | Classificação | 13/nov (entregue em 17/nov, com autorização do professor) | 60% | [Classificação](classification/index.md) |
| 3 | Generativo | 20/nov | 20% | [Generativo](generative/index.md) |

!!! danger "A nota do projeto é limitada pela prova de projeto (19/nov)"

    Nota da equipe = mínimo entre o projeto e a prova de projeto, que é uma nota **de
    equipe**: uma equipe que não sustenta o próprio trabalho na prova não fica com a nota
    que as entregas renderam. Os dois integrantes precisam conseguir explicar cada número e
    cada decisão — o raciocínio de cada entrega está em [Raciocínio](../raciocinio/index.md).

## Dataset

| | |
|---|---|
| **Nome** | Adult Income (Census Income, 1994) |
| **Fonte (URL)** | [UCI Machine Learning Repository — Adult](https://archive.ics.uci.edu/dataset/2/adult){:target='_blank'} (arquivo `adult.data`, versionado em `docs/projects/data/`). É o mesmo conjunto listado pelo professor como [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census){:target='_blank'} |
| **Licença / termos de uso** | CC BY 4.0 (UCI) |
| **Amostras** | 32561 (32537 após remover 24 duplicatas exatas) |
| **Features** | 14 no arquivo: 6 numéricas e 8 categóricas (12 após o EDA: 5 + 7) |
| **Variável alvo** | `income` — renda anual acima de US$ 50 mil (`>50K`, 24.09% após a limpeza) ou não (`≤50K`) |
| **Tarefa escolhida** | **Classificação** binária |

**Por que este dataset.** Tem tudo o que torna o pré-processamento para uma rede neural não
trivial — tipos misturados, faltantes que carregam informação, uma cauda extrema com valores
no teto (`capital-gain` = 99999), categorias raras e de alta cardinalidade, colunas
redundantes e um desbalanceamento moderado que impede a acurácia de ser uma métrica honesta.
E a classificação não é trivial: o melhor classificador "burro" (sempre `≤50K`) já acerta
75.91%, então o modelo da próxima entrega precisa superar esse piso de forma mensurável.

## Status

- [x] **1. EDA** — entregue em 08/out
- [x] **2. Classificação** — entregue em 17/nov (repositório migrado para o fork
  [`LagoaGustavo/ann-dl`](https://github.com/LagoaGustavo/ann-dl), mesmo histórico)
- [ ] **3. Generativo**

## Registro de decisões

| Data | Decisão | Motivo |
|------|---------|--------|
| 08/out | Dataset Adult Income, tarefa de classificação | Um dos quatro perfis sugeridos no enunciado; tipos misturados e desafios reais de pré-processamento (ver [Síntese do EDA](eda/index.md#5-synthesis)) |
| 08/out | Descartar `education` e `fnlwgt` | `education` é idêntica a `education-num` (bijeção de 16 níveis); `fnlwgt` é peso amostral do censo, sem relação com o alvo |
| 08/out | Faltantes de `workclass`/`occupation` viram a categoria `"Unknown"`, não a moda | A ausência é informativa: só 10.40% dessas pessoas ganham >50K, contra 24.08% no geral |
| 08/out | `log1p` em `capital-gain`/`capital-loss`, sem remover linhas | A regra do IQR marcaria como outlier toda pessoa com ganho de capital — 21.3% da classe positiva do treino |
| 08/out | `capital-gain` mantido, mas sinalizado como vazamento parcial | ≥ 7000 implica >50K em 98.6% dos casos; a próxima entrega mede o modelo com e sem ele |
| 17/nov | Repositório migrado para o fork `LagoaGustavo/ann-dl` | Perdemos o acesso à conta `@Lagoass`; o fork preserva o histórico e as datas das entregas anteriores |
| 17/nov | Validação = 20% do treino do EDA (20823 / 5206), pipeline ajustado só no sub-treino | O teste do EDA é usado uma única vez; todas as escolhas precisam de um conjunto que não seja o teste |
| 17/nov | Métrica principal PR-AUC; critério: logística + 0.01 e > dp entre seeds; custo C_FN = 3·C_FP | 24.09% de positivos: a acurácia não discrimina; escrito antes de treinar (commit `0b77e07`) |
| 17/nov | Rede final 50 → 64 → 1, ReLU, dropout 0.3, lr 3e-3, `pos_weight` 3.15 | Maior PR-AUC de validação entre 27 configurações (0.7773 ± 0.0039); congelada antes do teste |
| 17/nov | Resultado: MLP 0.7964 ± 0.0031 no teste, contra 0.7517 da logística; o HGB faz 0.8409 | Critério atingido, mas o boosting é melhor em dado tabular, e isso é dito no relatório |
