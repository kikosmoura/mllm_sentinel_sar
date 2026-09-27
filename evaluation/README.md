# Comparacao dos classificadores

Somente leitura dos modelos e predicoes existentes: estes comandos nao
treinam RF/CNN/MLLM nem executam inferencia ou nova ablation.

## Comparacao principal com ResNet-18

```bash
venv/bin/python 09_compare_models.py --include-cnn
venv/bin/python 11_generate_final_results.py --include-cnn
```

Ambos usam `results/final_with_resnet18/` por padrao. A representacao e fixada
em `pseudo_rgb` para a CNN e os MLLMs; RF mantem `vv_vh_features`.
Use `--cnn-results`/`--cnn-config` para indicar artefatos em outras pastas.
`--output-dir results/final` com CNN e rejeitado para preservar a consolidacao
anterior. Os dois passos precisam usar a mesma seed e numero de replicas
bootstrap (padrao 42 e 1000).

## Modo antigo, sem CNN

Sem `--include-cnn`, continuam sendo comparados os tres modelos anteriores.
Para executar sem modificar a consolidacao antiga, indique outra saida:

```bash
venv/bin/python 09_compare_models.py --output-dir /tmp/comparison_without_cnn
venv/bin/python 11_generate_final_results.py --output-dir /tmp/comparison_without_cnn
```

Nenhuma dependencia da CNN e importada nesse modo. O helper de proveniencia
usa apenas a biblioteca padrao e as funcoes de avaliacao existentes; nao abre
o checkpoint com PyTorch nem baixa pesos.

## Formatos e validacoes

`model_comparison.csv` mantem os nomes internos `Random Forest`,
`MLLM Zero-Shot`, `MLLM Fine-Tuned` e acrescenta `ResNet-18`.
As tabelas e figuras apresentam `MLLM Fine-Tuned` como **MLLM + LoRA**.
`Fine-Tuned` nos campos das tabelas da ablation tambem significa LoRA.

O segundo passo verifica que os valores da comparacao correspondem as
predicoes e aos ICs calculados com a configuracao solicitada. A condicao
MLLM pseudo-RGB deve ter exatamente as mesmas predicoes na comparacao
principal e na ablation existente. A CNN exige treinamento/avaliacao
completos, hashes dos tres splits e do checkpoint corretos, representacao
e pre-processamento correspondentes e zero uso de teste em treino/selecao.

`all_test_predictions.csv` tem RF + CNN + as seis condicoes existentes dos
MLLMs: oito condicoes, 320 linhas no teste atual de 40 imagens. As condicoes
MLLM pseudo-RGB aparecem uma unica vez, com o nome interno legado `MLLM`.
Sem CNN, sao sete condicoes e 280 linhas. Duplicatas por familia de modelo,
treinamento, representacao e imagem sao rejeitadas.

`table_main_results.csv` apresenta a visao `strict`, metricas, taxa de
predicoes invalidas, ROC-AUC e ICs de F1/Balanced Accuracy. As duas visoes,
`strict` e `valid-only`, continuam em `model_comparison.csv`.
ROC-AUC ausente e `NaN`; rotulos textuais nao sao convertidos em probabilidades.

`table_training_regimes.csv` e `experiment_manifest.json` distinguem
pre-treinamento, parametros e orcamentos de adaptacao. Valores nao
registrados ou nao aplicaveis ficam vazios/nulos. O total do MLLM com LoRA
inclui adapters; o total exato do zero-shot nao foi registrado separadamente.
Esses regimes nao sao apresentados como equivalentes.

O teste possui 40 imagens de Somalia e Sri-Lanka. Os ICs atuais sao bootstrap
por imagem, e nao por evento geografico. Diferencas pontuais e ICs por modelo,
isoladamente, nao demonstram superioridade estatistica. O campo da melhor
representacao pelo F1 de teste e mantido somente como descricao exploratoria
posterior; nunca seleciona a representacao principal.

```bash
venv/bin/python -m unittest discover -s tests -p 'test_model_comparison.py' -v
```
