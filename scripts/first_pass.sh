cd ..

export config=normal_gcn_bert_optuna
export hparams=normal_gcn_bert_hparams

# export features=avg#times#pair_encode#minus#concat
export features=avg#times#pair_encode#minus#concat
export dataset=matres_qiangning
export normal_epochs=30
export normal_bsz=16

# nghuyong/ernie-2.0-en, bert-base-uncased, roberta-base
export bert=bert-base-uncased
export bert_size=768

export classifier_type=hdn

allennlp tune \
    training_config/${config}.jsonnet \
    training_config/${hparams}.json \
    --include-package temporal \
    --serialization-dir experiments_v2/${config}_${dataset} \
    --metrics best_validation_fscore \
    --direction maximize
