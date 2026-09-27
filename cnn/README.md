# ResNet-18 para WATER/NON_WATER

Baseline de transferencia de aprendizado para classificar **o chip inteiro**
usando somente pixels dos PNGs SAR existentes. `NON_WATER=0`, `WATER=1`.
A comparacao com RF/MLLM permanece nos scripts atuais; esta implementacao
produz as predicoes no formato aceito por `evaluation/metrics.py`.

## Ambiente e comandos

Execute na raiz do repositorio com o ambiente existente. Dependencias:
PyTorch, Torchvision, Pillow, NumPy, scikit-learn e Matplotlib. Nao e
necessario atualizar o ambiente. CUDA e usada quando disponivel; caso
contrario, `--device auto` escolhe CPU. `--device cuda` indisponivel causa erro.

Treino principal (3 epocas de aquecimento e ate 25 epocas totais):

```bash
venv/bin/python 12_train_resnet18.py
```

Depois de completar o treino, avaliar o checkpoint escolhido:

```bash
venv/bin/python 13_evaluate_resnet18.py
```

Avaliacao somente na validacao, em outra pasta:

```bash
venv/bin/python 13_evaluate_resnet18.py \
  --splits validation --results-dir results/resnet18_pseudo_rgb_validation_only
```

Treinar somente `fc`, sem ajuste convolucional, em pastas distintas:

```bash
venv/bin/python 12_train_resnet18.py --mode head-only \
  --model-dir models/resnet18_pseudo_rgb_head_only \
  --results-dir results/resnet18_pseudo_rgb_head_only
venv/bin/python 13_evaluate_resnet18.py \
  --checkpoint models/resnet18_pseudo_rgb_head_only/best_checkpoint.pt \
  --results-dir results/resnet18_pseudo_rgb_head_only
```

`--representation vv` ou `vh` no treino escolhe o PNG correspondente e
altera os diretorios padrao para `resnet18_vv`/`resnet18_vh`. Na avaliacao,
indique `--checkpoint models/resnet18_vv/best_checkpoint.pt` (ou `vh`);
a representacao e o pre-processamento sao carregados do checkpoint.
`--project-root`, `--train`, `--validation`, `--test`, taxas de aprendizado,
batch size, threads e workers podem ser configurados; veja `--help`.
Os hashes impedem avaliar splits diferentes daqueles registrados no treino.
Novas execucoes exigem diretorios de saida diferentes para preservar artefatos.

Para nao abrir nem mesmo `test.csv` durante o treino, persista previamente o
retorno de `cnn.data.audit_splits` em JSON e passe `--split-audit ARQUIVO`.
Nesse modo, o treino le os metadados da auditoria, reconfere apenas os CSVs de
treino/validacao e deixa a reconferencia do teste para a avaliacao, apos fixar
o checkpoint. A origem e o hash dessa auditoria ficam na configuracao salva.

## Protocolo

- Pesos explicitos `ResNet18_Weights.IMAGENET1K_V1`, URL oficial e SHA256
  registrados. Falha de download/cache invalido causa erro, sem fallback.
- Aquecimento: treinar apenas `fc`; depois, apenas `layer4+fc`. Blocos anteriores
  permanecem em `eval`. Todas as estatisticas BatchNorm permanecem congeladas
  em ambas as fases. Os parametros afins de BatchNorm de `layer4` podem aprender.
- AdamW com `lr_fc=1e-3`, `lr_layer4=1e-4`, `weight_decay=1e-4`.
  O estado do otimizador e reiniciado na transicao de fase.
- Pesos de classe `N/(2*N_classe)` calculados somente em `train`.
  Cross entropy ponderada; loss agregada pela soma dos pesos dos alvos.
- Melhor checkpoint global, incluindo aquecimento, por Balanced Accuracy
  de validacao; desempate pela menor loss de validacao.
  Early stopping apos o aquecimento: 5 epocas consecutivas sem melhora.
- PNG RGB inteiro redimensionado para 224x224 com interpolacao bilinear,
  conversao para [0,1] e normalizacao ImageNet. Sem recortes ou alteracao de cor.
  Flips horizontal/vertical (probabilidade 0,5 cada) somente em `train`.
- Auditoria de todos os pares de splits por IDs e eventos. Durante o treino,
  o CSV de teste fornece apenas hash opaco e colunas `image_id/event/split`;
  nao se interpretam rotulos nem caminhos de imagens de teste. Os unicos
  datasets carregados pelo treinador sao `train` e `validation`.
- Caminhos antigos: preferir `PROJECT_ROOT/data/images/REPRESENTATION/ID.png`.
  Se a copia declarada tambem existir, exigir SHA256 identico. Registrar todas
  as copias verificadas; divergencia ou PNG invalido causa erro. Nenhum CSV/PNG
  e reescrito, nenhum exemplo e excluido. Hashes sao reconferidos no fim.
- Inferencia em ordem do CSV, sem augmentation, com argmax. Softmax float64;
  `prob_non_water=1-prob_water`, exportadas sem arredondamento intencional,
  verificadas pelo leitor compartilhado (tolerancia de soma 1e-9).
- Metricas e ICs usam exclusivamente `evaluation/metrics.py`, com as visoes
  `valid-only` e `strict`, classe positiva WATER e 1000 replicas bootstrap
  por padrao na avaliacao. Falhas nao viram predicoes inventadas.

O cache padrao e `.cache/resnet18/` na raiz do projeto (nao versionar os pesos).
Para funcionar offline, copie o arquivo oficial `resnet18-f37072fd.pth` para
`CACHE/checkpoints/` e use `--weights-cache CACHE` no treino.
O prefixo SHA256 do arquivo oficial e conferido mesmo quando ele ja esta em cache.
A avaliacao restaura o checkpoint integralmente sem baixar pesos ImageNet.

## Artefatos

- `models/resnet18_pseudo_rgb/best_checkpoint.pt`: estado completo, epoca,
  fase selecionada e configuracao/pre-processamento.
- `models/resnet18_pseudo_rgb/training_config.json`: origem dos pesos,
  hashes dos tres splits e do checkpoint, isolamento, resolucao dos PNGs de
  treino/validacao, contagens por classe, parametros por fase, seed, ambiente,
  criterio de selecao, tempos e `test_samples_used_for_training=0`.
  `status=complete` so aparece ao encerrar e verificar o treino.
- `results/resnet18_pseudo_rgb/training_history.csv` e `training_summary.json`.
- Avaliacao: `validation_predictions.csv`, `test_predictions.csv`, `metrics.csv`,
  `confusion_matrix_validation.png`, `confusion_matrix_test.png` e
  `evaluation_manifest.json`, conforme os splits solicitados.

Esses artefatos registram regimes e recursos; pre-treinamento ImageNet e treino
parcial da CNN nao equivalem ao pre-treinamento/LoRA dos MLLMs. O teste atual
tem 40 imagens de dois eventos; os ICs compartilhados sao por imagem.

## Verificacao curta

```bash
venv/bin/python -m unittest discover -s tests -p 'test_resnet18.py' -v
venv/bin/python -m cnn.smoke --weights-cache /tmp/resnet18-verification/weights
```

O smoke test carrega os pesos oficiais, realiza **dois passos** de otimizacao
(um por fase) com duas imagens reais de treino, verifica gradientes, blocos
congelados e estatisticas BatchNorm, restaura um checkpoint, exporta duas
predicoes reais de validacao, calcula metricas/ICs e salva uma matriz de confusao.
Todo artefato fica numa pasta temporaria removida ao final. O cache dos pesos
pode permanecer em `/tmp`; nao sao gerados resultados oficiais nem carregadas
imagens/rotulos de teste. Sem `--weights-cache`, o cache tambem e temporario.

As regressoes usam fixtures sinteticas em pastas temporarias e uma arquitetura
explicitamente sem pesos pre-treinados para testar regras de isolamento,
falhas, fases e ciclo de persistencia. Isso nao substitui a verificacao com os
pesos oficiais nem produz resultados do experimento.
