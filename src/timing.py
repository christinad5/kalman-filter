#!/usr/bin/env python3

import rospy
import numpy as np
import time
import quaternion
from std_msgs.msg import String, Float64, Float32, Int32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

class Duration:

    def __init__(self):
        self.time_prev = 0
        self.time_curr = 0
        self.time_diff = 1/13.32
        self.first_loop = True

    def callback(self, data):
        if self.first_loop == True:
            timing = data.header.stamp
            self.time_curr = timing.to_sec()
            self.first_loop = False
            print('first loop')
        else:
            self.time_prev = self.time_curr
            timing = data.header.stamp
            self.time_curr = timing.to_sec()
            self.time_diff = self.time_curr - self.time_prev
            print('previous time: ', self.time_prev)
            print('current time: ', self.time_curr)
            print(self.time_diff)

    def main(self):
        rospy.init_node('time_diff', anonymous=True)
        rospy.Subscriber("/sensor/Imu", Imu, self.callback)
        rospy.spin()


if __name__ == '__main__':
    try:
         duration = Duration()
         duration.main()
    except rospy.ROSInterruptException:
        pass

