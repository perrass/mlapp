# ML-Model-XGBoost

## Parameters

* colsample_bylevel: 0.88, 0.85. 每一次构建树时选择的特征数
* colsample_bytree: 0.77, 0.68. 每一次分离时选择的特征数. (如果这个数小对应的max_depth会高)
* subsample: 0.82, 0.66. 每一次构建树时选择的样本数
* gamma: 0.36, 0.74. Or min_split_loss, 如果分裂出的叶节点缩减小的loss小于这个值，则不分裂. 越大越保守
* learning_rate: 0.085, 0.07. 学习率
* max_depth: 19, 19. 树的最大深度，控制complexity
* min_child_weight: 7, 1. 在树分裂的那步，如果叶节点的sum of instance weight小于它，那么不会进行下一步的分裂. 越大越保守
* objective: logistic
* reg_alpha: 0.07, 0.47: l1 on weights
* reg_lambda: 0.53, 0.66: l2 on weights

 