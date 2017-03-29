# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import sys


# 教師データ（入力）
trainInput = np.array([
    [0., 0.],
    [1., 0.],
    [0., 1.],
    [1., 1.]
])

# 教師データ（出力）
trainOutput = np.array([
    0.,
    1.,
    1.,
    0.
])

# 教師データの総数
trainDataLen = len(trainInput)

# バッチサイズ（一回のセッション実行で利用する教師データの個数）
batchSize = 50

# ステップサイズ（1エポック内でのセッション実行回数）
stepSize = 100

# エポック・学習回数（エポック毎にパラメータ調整したりする）
epochSize = 10

# 正解値を返す関数
def xorGate(x, w, b):
    res = []

    for i in range(batchSize):
        wx0 = w[0] * x[i][0]
        wx1 = w[1] * x[i][1]
        res.append(abs(wx0 + wx1 + b))

    return res

# 入力プレースホルダ
x = tf.placeholder(tf.float32, name='x', shape=[batchSize, 2])

# 出力プレースホルダ
y = tf.placeholder(tf.float32, name='y', shape=[batchSize,])

# 重み
wVal1 = np.random.rand() * 10
wVal2 = np.random.rand() * 10
print('Init : w = [{0}, {1}]'.format(wVal1, wVal2))
w = tf.Variable([wVal1, wVal2])

# バイアス
bVal = np.random.rand() * 10
print('Init : b = {0}'.format(bVal))
b = tf.Variable(bVal)

# 正解値（プレースホルダの演算結果なのでこの時点で値は不定）
correctY = xorGate(x, w, b)

# 誤差関数（差の平方の平均）
costOp = tf.reduce_mean(tf.square(correctY - y))

# 最適化手法（勾配降下法）
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(costOp)

print('----------------')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochSize):
        indexes = np.random.randint(0, trainDataLen, batchSize)
        finalCost = 0
        for step in range(stepSize):
            optVal, cost = sess.run(
                [optimizer, costOp],
                feed_dict={
                    x : trainInput[indexes],
                    y : trainOutput[indexes]
                }
            )
            finalCost += cost
        print('finalCost at epoch_{0:0>4} : {1}'.format(epoch, finalCost))
    print('--- [Result] ---')
    w_res = sess.run(w)
    b_res = sess.run(b)
    wx0 = w_res[0] * trainInput[0][0]
    wx1 = w_res[1] * trainInput[0][1]
    print('False xor False = ' + str(abs(wx0 + wx1 + b_res)))
    wx0 = w_res[0] * trainInput[1][0]
    wx1 = w_res[1] * trainInput[1][1]
    print('True  xor False = ' + str(abs(wx0 + wx1 + b_res)))
    wx0 = w_res[0] * trainInput[2][0]
    wx1 = w_res[1] * trainInput[2][1]
    print('False xor True  = ' + str(abs(wx0 + wx1 + b_res)))
    wx0 = w_res[0] * trainInput[3][0]
    wx1 = w_res[1] * trainInput[3][1]
    print('True  xor True  = ' + str(abs(wx0 + wx1 + b_res)))
    print('----------------')
    #tf.summary.FileWriter('./', graph=sess.graph)

