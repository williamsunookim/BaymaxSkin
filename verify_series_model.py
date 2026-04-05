'''
TODO: 
callback 주기를 21hz로 고정시켜야됨
'''
from collections import deque
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import JointState
import numpy as np
# from sensor_msgs.msg import PointCloud2, PointField
# from std_msgs.msg import Header
# from ament_index_python import get_package_share_directory
import torch
from torch import nn

DEQUE_SIZE=2

class jointPointcloud(nn.Module):
    def __init__(self):
        super(jointPointcloud, self).__init__()

        # 8 * 20 -> 5568 with Batch Normalization

        self.decoder = nn.Sequential(
            nn.Linear(8*DEQUE_SIZE, 8),
            nn.BatchNorm1d(8),
            nn.SiLU(),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Linear(1024, 5568)
        )

    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

