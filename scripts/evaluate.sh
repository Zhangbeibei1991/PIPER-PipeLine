# evaluate.sh可以修改一些默认的参数，如log各个类型的PRF，绘制混淆矩阵，使用ILP等...

OUT=experiments_v2/normal_gcn_bert_optuna_tbd/trial_17  # 模型路径
DATA=data/matres_qiangning  # 数据路径
cd ..
allennlp evaluate \
${OUT}/model.tar.gz \
${DATA}/test.json \
--cuda-device 0 \
--include-package temporal \
--output-file ${OUT}/METRICS.json