#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step1. 简单神经网络
需求：
通过两层神经网络实现十分类任务
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data
from dependence import input_data

# 下载数据集
mnist = input_data.read_data_sets('data', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels
print('MNIST loaded')
print()

# 拓扑结构
# n_hidden_1: 第一层256个神经元
# n_hidden_2: 第二层128个神经元
# n_input: 输入像素点个数
# n_classes: 最终得出的分类类别(十分类)
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

# 输入和输出
# 样本个数不确定,用None
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

# 拓扑参数
# 指定一个初始化项: 指定方差项
stddev = 0.1
# 权重初始化: w1和w2进行高斯初始化
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print('NETWORK READY')











# trainimg.shape: (55000, 784) 表示训练样本有55000张, 784表示图像大小为28*28共784个像素点
# trainlabel.shape: (55000, 10) 表示55000个label对应55000个样本, 每个label有10个指标
# trainlabel[0]: [0,0,0,0,0,0,0,1,0,0], 属于该类别为1, 不属于为0, 此处第七个为1, 故该样本为7
print(trainimg.shape)
print(trainlabel.shape)
print(testimg.shape)
print(testlabel.shape)
# print(trainlabel[0])
print()

# 初始化变量
# x: [None, 784] 行:None在tf中表示无穷,因为目前不知道有多少个样本 列:每张图片784个像素点
# y: [None, 10] 行:None在tf中表示无穷,因为目前不知道有多少个样本 列:本次使用十分类
# W: [784, 10] 行:对于784个像素点,必须都跟权重参数W相乘(矩阵相乘第一个矩阵的列数等于第二个矩阵的行数) 列:即输出多大,然后得到当前输入等于10类值里的哪个
# b: 本次做的是十分类任务,故初始化10个b
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
# 建立逻辑回归模型
# softmax将二分类升级成多分类
# Wx+b得到一维矩阵(10个值),表示当前样本在0~9十个类别中每个类别的分值大小
actv = tf.nn.softmax(tf.matmul(x, W) + b)
# 计算损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(actv), reduction_indices=1))
# 指定优化器
# learning_rate: 学习率
# GradientDescentOptimizer: 梯度下降优化器
# .minimize(cost): 最小化cost
learning_rate = 0.1
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
print()

# 预测与真实值对比(prediction): 预测值的索引和结果值索引一样时返回true,否则false
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# 求准确率(accuracy): 将预测值(true/false)转化为float(1/0), 然后求所有的均值
accr = tf.reduce_mean(tf.cast(pred, 'float'))
# initializer
init = tf.global_variables_initializer()

# 所有样本迭代50次
training_epochs = 50
# 每次迭代选择100个样本
batch_size = 100
# 每5个epoch打印一下
display_step = 5

# 创建session
sess = tf.Session()
sess.run(init)
# mini-batch learning
for epoch in range(training_epochs):
    avg_cost = 0
    num_batch = int(mnist.train.num_examples / batch_size)
    for i in range(num_batch):
        # batch_xs:一个batch的data, batch_ys:一个batch的label
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 通过梯度下降迭代一次
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
        feeds = {x: batch_xs, y: batch_ys}
        # 计算损失值: 打印用
        avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
    # display
    if epoch % display_step == 0:
        feeds_train = {x: batch_xs, y: batch_ys}
        feeds_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feeds_train)
        test_acc = sess.run(accr, feed_dict=feeds_test)
        print("epoch: {epoch}/{training_epochs}, cost: {cost}, train_acc: {train_acc}, test_acc: {test_acc}".format(
            epoch=epoch, training_epochs=training_epochs, cost=avg_cost, train_acc=train_acc, test_acc=test_acc
        ))
print('Done')
