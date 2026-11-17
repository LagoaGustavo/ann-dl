# Raciocínio — Projeto, entrega 2: Classificação

Como pensamos a [classificação do Adult Income](../projects/classification/index.md), na
ordem em que as decisões apareceram. A prova de projeto é nota **de equipe** e limita a
nota do projeto: os dois integrantes precisam conseguir refazer cada passo a partir daqui.

## Etapa 0 — Ler o enunciado caçando o que custa nota

Cinco regras mandam em todo o resto:

- **Mesmo dataset, mesma tarefa, mesmo split e mesma seed do EDA.** Por isso a tarefa é
  classificação. Não havia escolha: o EDA declarou `task: classification`, e o Adult não
  tem alvo contínuo. O split também é o do EDA: importamos o `pipeline.py`, sem copiar.
- **O teste é usado uma única vez.** Se ele ajudar em qualquer escolha, o passo 5 vale no
  máximo metade. Daí a ordem dos commits: critério → baselines → busca → **congelar a
  configuração** → só então o script do teste. Os commits registram a **ordem** das etapas,
  não o horário: todos foram gravados em 17/nov. A garantia real está no código, porque o
  script do teste só **lê** a configuração congelada e não escolhe nada.
- **A métrica e o critério ficam escritos antes de treinar.** Estão em `comum.py` desde o
  primeiro commit da entrega (`0b77e07`).
- **A saída é um logit e a perda é `BCEWithLogitsLoss`**, com `pos_weight` para o
  desbalanceamento.
- **O limiar é escolhido na validação e justificado pelo custo dos erros**, não pelo
  "máximo F1".

## Etapa 1 — A validação sai de dentro do treino

O EDA tinha só treino (26029) e teste (6508). Tiramos 20% do treino, estratificado e com a
mesma seed: 20823 de sub-treino e 5206 de validação. O pré-processador é ajustado **só no
sub-treino**. Se fosse ajustado no treino inteiro, as médias e desvios já teriam "visto" a
validação, que é onde fazemos todas as escolhas.

A prova de que nada vazou é a mesma do EDA: a média das numéricas padronizadas dá 0 no
sub-treino e ≠ 0 na validação.

## Etapa 2 — Métrica, critério e custo

- **PR-AUC** e não acurácia: o Dummy acerta 75.91% sem aprender nada. A PR-AUC tem piso
  na prevalência (0.241) e não depende de limiar.
- **Critério:** a rede precisa ter ≥ +0.01 de PR-AUC sobre o melhor baseline obrigatório,
  e a diferença precisa ser maior que o desvio entre seeds. Se "ganhar" com uma diferença
  menor que a variação de seed, não é ganho.
- **Custo 3:1** (FN custa 3 vezes o FP): é o cenário de triagem para uma oferta, em que
  perder um cliente de alta renda custa mais que abordar alguém à toa.

## Etapa 3 — Baselines: a barra de verdade é a logística

| Validação | PR-AUC |
|---|---|
| Dummy | 0.2409 |
| Logística | 0.7283 |
| HistGradientBoosting (opcional) | 0.8205 |

Já sabíamos ali que as árvores seriam difíceis de bater. O critério compara com a
logística, porque ela é o baseline obrigatório (o enunciado pede trivial + linear). O HGB
aparece em todas as tabelas para sermos honestos.

## Etapa 4 — A rede e o treino (uma surpresa)

A rede base era 50 → 64 → 32 → 1, com 5377 parâmetros. Ela treina em ~0.05 s por época.
A varredura de lr deu a menor perda de validação em 1e-3.

**Surpresa:** a perda de validação atinge o mínimo na época 11, mas a PR-AUC de validação
continua subindo até as épocas ~30–40 (0.7727 na 26). A perda (log-loss) pune a confiança excessiva antes de a
ordenação piorar. Parar pela perda (como no esqueleto do enunciado) é conservador. Mantivemos
a parada pela perda e registramos a discordância.

## Etapa 5 — Regularização: um fator por vez, depois a busca

- Weight decay sozinho quase não faz nada. Com lr 1e-3, encolhe os pesos menos de 1% por
  época.
- Dropout segura o overfitting (a perda de validação não sobe), mas sozinho não melhora a
  PR-AUC na rede de 2 camadas.
- BatchNorm **piorou**.
- Weight decay + dropout foi o melhor da parte A.

A busca (27 configurações × 3 seeds) escolheu **uma camada de 64, lr 3e-3, dropout 0.3**
(PR-AUC de validação 0.7773 ± 0.0039). Redes maiores não ajudaram. Congelamos essa
configuração num commit antes de criar o script do teste.

## Etapa 6 — O teste, uma vez

- MLP: **0.7964 ± 0.0031**, contra 0.7517 da logística. O critério foi atingido
  (+0.0447, cerca de 14 vezes o dp).
- HGB: **0.8409**. A rede não é o melhor modelo, e dissemos isso.
- Limiar ≈ 0.53, e não 0.25. O `pos_weight` ≈ 3.15 multiplica as **chances** p/(1−p) da
  saída por ~3.15. Então o limiar de custo 3:1 (0.25 real, chance 1/3) vira chance ≈ 1.05
  na saída, ou seja, ≈ 0.51.

## Etapa 7 — Erros

- Entre os subgrupos com n ≥ 100, o maior erro é em `relationship = Husband` (32.5%),
  quase todo FP. Os casados têm
  prevalência ~45%, perto da indecisão, e o custo 3:1 empurra o limiar a favor de marcar
  `>50K`.
- Mulheres, pessoas negras e jovens têm FNR alto: a rede aprendeu as taxas-base de 1994.
- Na permutação, `relationship` cai de 1º (no EDA) para 8º, porque é redundante com
  `marital-status` + `sex`.
- Sem capital, a PR-AUC cai 0.066, mas a rede continua melhor que a logística.
- Sem `sex`/`race`/`relationship`, não se perde nada.

## Perguntas que esperamos na prova

| Pergunta | Resposta curta |
|---|---|
| Por que a validação não é o teste do EDA? | Porque o teste só pode ser usado uma vez. Todas as escolhas (lr, arquitetura, regularização, limiar, época) são feitas na validação, que sai do treino. |
| Por que ajustar o pipeline só no sub-treino? | Para que nenhuma estatística (média, desvio, mediana, categorias raras) venha da validação, onde se fazem as escolhas. A prova é a média 0 só no sub-treino. |
| O que é o `pos_weight` e quanto vale? | O peso dos positivos na BCE, `N_neg/N_pos` do sub-treino: 15806/5017 = 3.15. Cada `>50K` pesa 3.15 vezes um `≤50K` no gradiente. |
| Por que a rede devolve logit e não probabilidade? | `BCEWithLogitsLoss` aplica a sigmoide dentro da perda com o truque log-sum-exp, que é estável. Se a rede tivesse uma sigmoide no fim, a sigmoide seria aplicada duas vezes. O gradiente na saída continua σ(u) − y (ponderado), como no exercício MLP. |
| Quantos parâmetros tem a rede final? | 50·64 + 64 = 3264 na oculta, mais 64 + 1 = 65 na saída: **3329**, contra 20823 linhas (0.16 por linha). |
| Por que He e não Glorot? | He compensa a ReLU zerar metade das ativações (variância 2/fan_in). Na saída, que é linear, usamos Glorot. |
| Como funciona a parada antecipada? | A cada época mede a perda de validação. Se melhorou, guarda uma **cópia** (`deepcopy`) dos pesos. Depois de 15 épocas sem melhorar, para e recarrega a melhor cópia. |
| Por que AdamW se o weight decay final é 0? | Para a comparação do passo 4 ser limpa: com `weight_decay = 0` o AdamW é o Adam, e o mesmo otimizador serve para todas as variantes. O decaimento só vai nas matrizes de pesos. |
| Por que o limiar é 0.53 se o custo é 3:1? | Com probabilidades calibradas, o ótimo seria 0.25. Mas o `pos_weight` multiplica as **chances** da saída por ~3.15 (o logit sobe ln 3.15). 0.5 na saída equivale a 1/(1 + 3.15) ≈ 0.24 real, e 0.25 real equivale a ≈ 0.51 na saída. A varredura na validação acha 0.53 sem precisar de calibração. |
| A rede valeu a pena? | Contra a logística, sim (+0.045 de PR-AUC, cerca de 14 dp, e −7.7% de custo). Contra o boosting, não: o HGB ficou 0.0445 acima. Em produção, usaríamos árvores. |
| O teste influenciou alguma escolha? | Não. A configuração está congelada em `comum.py` (commit `f89dce2`), e o script do teste entra no commit seguinte (`1d4da6b`). Os commits mostram a ordem, não o horário. A garantia está no código: o `s56` só lê `CONFIG_ESCOLHIDA`, e o limiar e a época vêm da validação. |
| Qual o subgrupo de maior erro e por quê? | Entre os grupos com n ≥ 100, `Husband`, com 32.5%, quase tudo FP. A prevalência é de ~45% e o custo 3:1 empurra para marcar `>50K`. (`Married-AF-spouse` tem n = 6: é ruído.) |
| A permutation importance bate com o EDA? | Em boa parte (Spearman 0.72). Exceção: `relationship`, que é redundante com `marital-status` + `sex`. A permutação subestima features correlacionadas. |
