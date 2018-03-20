# Reguralized Selective ensemble

## データの用意

1. `train_test_split` → `data/data_n/` 以下に保存 n:0-4
    - 最終的な評価のための5fold
    - `Input: all.data`
    - `Output: data_n/X_test.npy X_train.npy y_test.npy y_train.npy`

2. `1` のtrainデータを5分割
    - `regularized selective ensemble` の `lambda` 決定のための5fold
    - `train_test_split` → `data/data_n/fold_n` 以下に保存 n:0-4
    - `Input: data_n/X_test.npy X_train.npy y_test.npy y_train.npy`
    - `Output: data_n/fold_n/X_test.npy X_train.npy y_test.npy y_train.npy`

3. アンサンブル候補モデルの予測を保存
    - `regularized selective ensemble` の `lambda` 決定のための5fold
    - `Input: data_n/fold_n/X_test.npy X_train.npy y_test.npy y_train.npy`
    - `Output: data_n/fold_n/predictions.csv truelabel.csv`

4. アンサンブル候補モデルの予測を保存
    - `3` と同じモデル群であること
    - `Input: data_n/X_test.npy X_train.npy y_test.npy y_train.npy`
    - `Output: data_n/predictions.csv truelabel.csv`

## rseの計算

5. `get_kernel_link_matrix.py` 実行
    - fold毎の `w_link` の計算
    - data毎の `w_link` の計算
    - `Input: data_n/fold_n/X_test.npy data_n/X_test.npy`
    - `Output: data_n/w_link/w_link_n.csv data_n/fold_n/w_link/w_link_n.csv`

6. `nohup echo " weight_for_train_data()" | matlab -nodisplay > weight.out &` 実行
    - train に対する `weight` の計算
    - `Input: data_n/fold_n/predictions.csv w_link/w_link_n.csv`
    - `Output: data_n/fold_n/weight/weight_lambda_n.csv`

7. `lambda_decision.py` 実行
    - data毎の `lambda` の決定
    - `Input: data_n/fold_n/predictions.csv weight/weight_lambda_n.csv`
    - `Output: lambda for each data`

8. `eval "matlab -nodesktop -nosplash -r \"weight_for_test_data($lambda)\" > weight.out"` 実行
    - train で決定した `lambda` に対して test に対する `weight` の計算
    - `Input: data_n/predictions.csv w_link/w_link_n.csv`
    - `Output: data_n/weight/weight_lambda_n.csv`

## 評価

9. 結果の確認
    - `result.py` でRSEの結果確認
