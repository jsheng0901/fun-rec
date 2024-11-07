import pandas as pd


def kaggle_loader(path):
    """
    数据读取与预处理
    """
    # 数据读取
    df_train = pd.read_csv(path + 'kaggle_train.csv')
    df_test = pd.read_csv(path + 'kaggle_test.csv')

    # 简单的数据预处理
    # 去掉id列， 把测试集和训练集合并， 填充缺失值
    df_train.drop(['Id'], axis=1, inplace=True)
    df_test.drop(['Id'], axis=1, inplace=True)

    df_test['Label'] = -1

    data = pd.concat([df_train, df_test])
    data.fillna(-1, inplace=True)

    # 下面把特征列分开处理
    numerical_fea = ['I' + str(i + 1) for i in range(13)]
    categorical_fea = ['C' + str(i + 1) for i in range(26)]

    return data, categorical_fea, numerical_fea
