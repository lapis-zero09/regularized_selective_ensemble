import glob
import sys
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


def parse_dir(dir_name):
    files = [f.split('/')[-1].replace('weight_lambda_', '').replace('.csv', '') for f in glob.glob('./{}/*.csv'.format(dir_name))]
    return files


ret_arr = []
name = 'data'
for data in range(5):
    files = parse_dir('../{}/data_{}/fold_0/weight/'.format(name, data))
    res = np.zeros((5, len(files)))

    for fold in range(5):
        p = pd.read_csv(
            '../{}/data_{}/fold_{}/predictions.csv'.format(name,
                                                              data, fold),
            header=None)
        p_nn = pd.read_csv(
            '../{}/data_{}/fold_{}/predictions_nn.csv'.format(name,
                                                                 data, fold),
            header=None)
        prd = pd.concat([p,
                         p_nn])[list(range(p.shape[1]))].reset_index(drop=True)
        print('fold_', fold)
        y_true = pd.read_csv(
            '../{}/data_{}/fold_{}/truelabel.csv'.format(name,
                                                            data, fold),
            header=None)
        y_true = np.where(y_true <= 0.0, 0, 1).flatten()

        for i, filename in enumerate(files):
            w = pd.read_csv(
                '../{}/data_{}/fold_{}/weight/weight_lambda_{}.csv'.format(name, data, fold, filename), header=None).T
            h = w.dot(prd)
            pred = np.where(h <= 0.0, 0, 1).flatten()
            mse = mean_squared_error(y_true, pred)
            res[fold, i] = mse
            print(filename, mse)

        print()

    res = pd.DataFrame(res, columns=files)
    means = res.mean()
    stds = res.std()

    d = {(std * 2, f): mean for f, (mean, std) in zip(files, zip(means, stds))}
    files = []
    for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True):
        print('%f (+/-%f) for %s' % (v, k[0], k[1]))
        files.append(k[1])

    # res.loc[:, files].plot.box(rot=90)
    # plt.tight_layout()
    # plt.savefig('../{}/data_{}/lambda.png'.format(name, data))
    ret_arr.append(float(k[1]))
    print(float(k[1]))

print(ret_arr)
