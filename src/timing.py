#!/usr/bin/env python3

import rospy
import numpy as np
import time
import quaternion
from std_msgs.msg import String, Float64, Float32, Int32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

def callback1(data):
    timing1 = data.header.stamp
    timing1 = float(timing1)
    timing_secs = data.header.stamp.secs
    timing_nsecs = data.header.stamp.nsecs
    rospy.loginfo('timing is %s', timing1)
    rospy.loginfo('timing seconds is %s', timing_secs)
    rospy.loginfo('timing nanoseconds is %s', timing_nsecs)

def time_diff(timing1, timing2):
    time_diff = timing2 - timing1
    return time_diff

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/sensor/Imu", Imu, callback1)
    rospy.spin()


if __name__ == '__main__':
    listener()
