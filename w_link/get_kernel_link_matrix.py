import subprocess
import time
import numpy as np
import sys
import array
from py4j.java_gateway import JavaGateway


def calc_w_link_on_java(data, gateway_inst, fold):
    # num_inst = data.shape[0]
    header = array.array('i', list(data.shape))
    body = array.array('d', data.flatten().tolist())
    if sys.byteorder != 'big':
        header.byteswap()
        body.byteswap()
    buf = bytearray(header.tostring() + body.tostring())
    java_matrix = gateway_inst.getKernelLinkMatrix(buf, fold)
    return java_matrix


name = 'data'
for data in range(5):
    print('[*] Starting data_{}'.format(data))
    for fold in range(5):
        args = (['java', '-Xmx4096g', '-cp',
                 'Extentions/weka.jar:Extentions/py4j.jar:./',
                 'GetKernelLinkMatrix'])
        p = subprocess.Popen(args)
        # サーバー起動前に処理が下へ行くのを防ぐ
        time.sleep(3)
        # JVMへ接続
        gateway = JavaGateway(start_callback_server=True)
        # GetKernelLinkMatrixのインスタンスを取得
        gateway_inst = gateway.entry_point
        fold_file = 'data_{}/fold_{}'.format(data, fold)
        print('\t\t[*] Starting {}'.format(fold_file))
        try:
            x = np.load('../{}/{}/X_test.npy'.format(name, fold_file))
        except:
            x = np.load('../{}/{}/X_test.npz'.format(name, fold_file))['d']
        w_link = calc_w_link_on_java(x, gateway_inst, fold_file)
        print(w_link)

        gateway.shutdown()
        print('\t\t[*] Success for shutdown Java...\n')

    args = (['java', '-Xmx4096g', '-cp',
             'Extentions/weka.jar:Extentions/py4j.jar:./',
             'GetKernelLinkMatrix'])
    p = subprocess.Popen(args)
    # サーバー起動前に処理が下へ行くのを防ぐ
    time.sleep(3)
    # JVMへ接続
    fold_file = 'data_{}'.format(data)
    gateway = JavaGateway(start_callback_server=True)
    # GetKernelLinkMatrixのインスタンスを取得
    gateway_inst = gateway.entry_point
    print('\t[*] Starting test data')
    try:
        x = np.load('../{}/{}/X_test.npy'.format(name, fold_file))
    except:
        x = np.load('../{}/{}/X_test.npz'.format(name, fold_file))['d']
    w_link = calc_w_link_on_java(x, gateway_inst, fold_file)
    print(w_link)

    gateway.shutdown()
    print('\t[*] Success for shutdown Java...\n')
    print('\t[-] Done data_{}'.format(data))
    print()
