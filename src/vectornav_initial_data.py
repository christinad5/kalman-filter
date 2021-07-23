#!/usr/bin/env python3

"""This code publishes and prints vectornav orientation data to initiate the EKF algorithm"""

import numpy as np
import math

import rospy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from geometry_msgs.msg import Quaternion

quat_real = [1, 0, 0, 0]

def callback_orientation(data):
    global quat_real
    quat_real = np.array([data.orientation.w, data.orientation.x, data.orientation.y, 
    data.orientation.z])


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/sensor/Imu", Imu, callback_orientation)
    
    pub_quaternion_transform = rospy.Publisher('/initial_quat', Quaternion, queue_size=10)
    rate = rospy.Rate(10)
    
    while not rospy.is_shutdown():
        q = Quaternion()
        q.w = quat_real[0]
        q.x = quat_real[1]
        q.y = quat_real[2]
        q.z = quat_real[3]

        rospy.loginfo(q)
        pub_quaternion_transform.publish(q)
    rospy.spin()


if __name__ == '__main__':
	listener()
