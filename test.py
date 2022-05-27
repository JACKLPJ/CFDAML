#加载函数包
import pandas as pd
import numpy as np
import AutoML8w_stack as automl
import sklearn
#import time 
# 设置软件参数
CFDAML = automl.Automl(data_pre_processing=True,# 是否需要数据预处理 (可选：True,False；默认：False)
                       system='linux',# 系统型号（可选：'linux','windows','mac'；默认：'linux'）
                       N_jobs=-1,# 并行运行的CPU核数 (默认：-1（表示使用机器所有CPU核）)
                       verbose=False,# 是否显示软件运行的中间结果 (可选：True,False；默认：False)
                       time_per_model=360# 训练单个模型管道的时间上限（默认：360（秒））
                      ) 
# 读取待测数据集并分裂训练集和测试集
publishing_data = pd.read_csv(
    '/media/sia1/Elements SE/AutoML测试数据集/DataSets/publishing_data.csv',
    sep=',',
    header=None)
X, y = publishing_data.iloc[:, :9], publishing_data.iloc[:, 9]
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=42)
# 模型的训练和预测
#t0 = time.perf_counter() # 记录训练和测试全过程的时间
CFDAML.fit(X_train, y_train) # 模型的训练
y_hat = CFDAML.predict(X_test) # 模型的预测
#print("Runtime:",time.perf_counter() - t0) # 打印时间开销
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, y_hat)) # 打印测试集上的准确率