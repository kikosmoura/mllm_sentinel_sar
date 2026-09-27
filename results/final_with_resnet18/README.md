# Comparacao final

Modelos: Random Forest, CNN (ResNet-18), MLLM zero-shot, MLLM + LoRA.

A representacao principal e fixada em `pseudo_rgb` para os modelos de imagem; RF usa `vv_vh_features`. Classe positiva: `WATER`. A tabela principal usa a visao `strict`; `model_comparison.csv` preserva tambem `valid-only`. `MLLM Fine-Tuned` e `Fine-Tuned` nas tabelas de ablation significam **MLLM + LoRA**. Os nomes internos anteriores permanecem nos CSVs de avaliacao.

ROC-AUC dos MLLMs e indisponivel (`NaN`), pois suas respostas nao incluem probabilidades legitimas. Nenhuma probabilidade foi inferida de rotulos textuais.

O teste contem 40 imagens de 2 eventos geograficos (Somalia, Sri-Lanka). Os ICs de 95% sao bootstrap por imagem; nao representam incerteza entre eventos. Diferencas pontuais e esses ICs, isoladamente, nao demonstram superioridade estatistica entre modelos.

`table_training_regimes.csv` e o manifesto documentam pre-treinamento, parametros e orcamentos diferentes. Campos vazios/nulos significam nao aplicavel ou nao registrado. Nao se assume igualdade de dados externos, tempo ou adaptacao. O total de parametros do MLLM com LoRA inclui os adapters; o total exato do zero-shot nao foi registrado separadamente.

O consolidado tem 8 condicoes e 320 linhas. A ablation dos MLLMs em VV/VH/pseudo-RGB e reutilizada; suas condicoes pseudo-RGB nao sao duplicadas. Nenhum modelo foi retreinado e nenhuma ablation da CNN foi executada.

`best_finetuned_representation_by_test_f1` e apenas descricao exploratoria posterior ao teste, sem efeito sobre a comparacao principal.
