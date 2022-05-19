from __future__ import print_function
from platform import node
import sys


import numpy as np
from math import floor
import tensorflow as tf
import TensorFI as ti
from TensorFI import faultTypes as ft

node1 = tf.zeros([2,2,50,20])

node2 = tf.zeros([2,2,50,20])

node3 = tf.add(node1, node2, name = "add1")



node1=np.squeeze(node1)
print(np.shape(node3))
print(node1)

s = tf.Session()

# Run it first
res1 = s.run([ node3 ])
print("res1 = ", res1)

# Instrument the FI session 
fi = ti.TensorFI(s , logLevel = 100)

# Create a log for visualizng in TensorBoard
logs_path = "./logs"
logWriter = tf.summary.FileWriter( logs_path, s.graph )

# Run it again with fault injection enabled
res2=s.run(node3);
# res2 = s.run([ node3 ])
print("res2 = ", res2)