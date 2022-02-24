# -- coding : utf-8 --
# @Time:2022/1/25 14:08
# @Author: Jianing Gou(goujianing19@mails.ucas.ac.cn)
"""
this file is used to test if the gpu is available for tensorflow training.
"""
import tensorflow as tf

print(tf.__version__)

print('GPU', tf.config.list_physical_devices('GPU'))

a = tf.constant(2.0)
b = tf.constant(4.0)
print(a + b)

"""
out: 

"""
