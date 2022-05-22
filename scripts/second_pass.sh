cd ..

export config=flip_gcn_bert_optuna
export hparams=flip_gcn_bert_hparams

# export features=avg#times#pair_encode#minus#concat
export features=avg#times#pair_encode#minus#concat
export dataset=matres_qiangning
export flip_epochs=30
export flip_bsz=4

# nghuyong/ernie-2.0-en, bert-base-uncased, roberta-base
export bert=bert-base-uncased
export bert_size=768

export classifier_type=hdn

export gcn_num_layers=3  # 第一轮中选择好的超参数
export hidden_size=572  # 第一轮中选择好的超参数

export flip_pretrained_path=experiments_v2/normal_gcn_bert_optuna_tbd/trial_0


allennlp tune \
    training_config/${config}.jsonnet \
    training_config/${hparams}.json \
    --include-package temporal \
    --serialization-dir experiments_v2/${config}_${dataset} \
    --metrics best_validation_fscore \
    --direction maximize
