#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step1. 变量
"""
import tensorflow as tf

# 创建行向量
x = tf.Variable([[2.7, 3.7]])
# 创建列向量
y = tf.Variable([[1.0], [2.0]])
# 运算: 矩阵相乘
z = tf.matmul(x, y)

# 变量只有初始化后才能打印出值
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(z.eval())
