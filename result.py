import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
import glob


class CheckResult:
    def __init__(self, name='data'):
        self.name = name
        p = pd.read_csv('{}/data_0/predictions.csv'.format(self.name),
                            header=None)
        self.len = p.shape[0]
        
    def best_single_model(self):
        bst = np.zeros((5, self.len))
        bst_f1 = np.zeros((5, self.len))

        for fold in range(5):
            p = pd.read_csv('{}/data_{}/predictions.csv'.format(self.name, fold),
                            header=None)
            prd = np.where(p <= 0.0, 0, 1)
            # p_nn = pd.read_csv(
            #     '{}/data_{}/predictions_nn.csv'.format(self.name, fold),
            #     header=None)
            # prd = pd.concat([p, p_nn])[list(range(p.shape[1]))].reset_index(
            #     drop=True)
            # prd = np.where(prd <= 0.0, 0, 1)
            y_true = pd.read_csv(
                '{}/data_{}/truelabel.csv'.format(self.name, fold),
                header=None)
            y_true = np.where(y_true <= 0.0, 0, 1).flatten()

            for i in range(prd.shape[0]):
                mse = mean_squared_error(y_true, prd[i])
                bst[fold, i] = mse
                f1 = f1_score(y_true, prd[i])
                bst_f1[fold, i] = f1

        bst = pd.DataFrame(bst)
        bst_f1 = pd.DataFrame(bst_f1)
        # bst.describe()
        return sorted(bst.mean())[0], sorted(bst_f1.mean())[-1]

    def rse(self, fold, w):
        p = pd.read_csv('{}/data_{}/predictions.csv'.format(self.name, fold),
                        header=None)
        prd = np.where(p <= 0.0, 0, 1)
#         p_nn = pd.read_csv(
#             '{}/data_{}/predictions_nn.csv'.format(self.name, fold),
#             header=None)
#         prd = pd.concat([p, p_nn])[list(range(p.shape[1]))].reset_index(
#             drop=True)
#         prd = np.where(prd <= 0.0, 0, 1)
        y_true = pd.read_csv(
            '{}/data_{}/truelabel.csv'.format(self.name, fold),
            header=None)
        y_true = np.where(y_true <= 0.0, 0, 1).flatten()

        def _rse_w(prd, w, y_true):
            # rse_w
            ret = []
            for i in range(prd.shape[1]):
                k = 0 if prd[:, i].dot(np.array(w).flatten()) < 0.5 else 1
                ret.append(k)

            mse = mean_squared_error(y_true, ret)
            f1 = f1_score(y_true, ret)
            return mse, f1

        def _rse(prd, w, y_true):
            w = np.where(w <= 0.0, 0, 1).flatten()
            idx = np.where(w > 0)[0]

            # rse
            ret = []
            for i in range(prd.shape[1]):
                count = collections.Counter(prd[idx, i])
                k = sorted(count.items(), key=lambda x: x[1])[-1][0]
                ret.append(k)

            mse = mean_squared_error(y_true, ret)
            f1 = f1_score(y_true, ret)
            return mse, f1

        rse_w, rse_w_f1 = _rse_w(prd, w, y_true)
        rse, rse_f1 = _rse_w(prd, w, y_true)

        mse = rse if rse < rse_w else rse_w
        f1 = rse_w_f1 if rse_f1 < rse_w_f1 else rse_f1

        return mse, f1

    def ensemble(self):
        mse = []
        f1 = []
        for fold in range(5):
            f = glob.glob('{}/data_{}/weight/weight_lambda_*.csv'.format(self.name, fold))
            if f:
                w = pd.read_csv(f[0], header=None)
                mse_, f1_ = self.rse(fold, w)
                print('rse')

            mse.append(mse_)
            f1.append(f1_)
        print('mse:', mse)
        print('f1:', f1)
        return np.mean(mse), np.mean(f1)
