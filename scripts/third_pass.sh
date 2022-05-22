cd ..

export config=triplet_gcn_bert_optuna
export hparams=triplet_gcn_bert_hparams

# export features=avg#times#pair_encode#minus#concat
export features=avg#times#pair_encode#minus#concat
export dataset=matres_qiangning
export triplet_epochs=30
export triplet_bsz=16

# nghuyong/ernie-2.0-en, bert-base-uncased, roberta-base
export bert=bert-base-uncased
export bert_size=768

export classifier_type=hdn

export gcn_num_layers=2  # 第一轮选好的超参数
export hidden_size=731

export triplet_pretrained_path=experiments_v2/flip_gcn_bert_optuna_tbd/trial_44


allennlp tune \
    training_config/${config}.jsonnet \
    training_config/${hparams}.json \
    --include-package temporal \
    --serialization-dir experiments_v2/${config}_${dataset} \
    --metrics best_validation_fscore \
    --direction maximize
