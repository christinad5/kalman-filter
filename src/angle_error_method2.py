#!/usr/bin/env python3

import rospy
import numpy as np
import time
import quaternion
from std_msgs.msg import String, Float64, Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

#calculate angles between Vectornav quaternion

def imu_angle(quat1_imu, quat2_imu):
    q = quat1_imu * quat2_imu
    phi = np.arctan2((q.w*q.x + q.y*q.z), 1 - 2*(np.square(q.x)+np.square(q.y)))
    theta = np.arcsin(2*(q.w*q.y - q.z*q.x))
    psi = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(np.square(q.y)+np.square(q.z)))
    angles_rad = [phi, theta, psi]
    return angles_rad


def EKF_angle(quat1_EKF, quat2_EKF):
    q = quat1_EKF * quat2_EKF
    phi = np.arctan2((q.w*q.x + q.y*q.z), 1 - 2*(np.square(q.x)+np.square(q.y)))
    theta = np.arcsin(2*(q.w*q.y - q.z*q.x))
    psi = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(np.square(q.y)+np.square(q.z)))
    angles_rad = [phi, theta, psi]
    return angles_rad


def error_calc(angles_imu, angles_EKF):
    angle_error = []
    for i in range(len(angles_imu)):
        if angles_imu[i] != 0: 
            error = 100*np.abs(((angles_imu[i] - angles_EKF[i])/angles_imu[i]))
            angle_error.append(error)
        elif angles_imu[i] == 0:
            if angles_EKF[i] != 0:
                error = 100*np.abs(((angles_imu[i] - angles_EKF[i])/angles_EKF[i]))
                angle_error.append(error)
            elif angles_EKF[i] == 0:
                error = 0
                angle_error.append(error)
        else:
            angle_error.append('8888')
    return angle_error

class AngleClass:

    def __init__(self):
        self.quat1_imu = np.quaternion(1,0,0,0)
        self.quat2_imu = np.quaternion(1,0,0,0)
        self.quat1_EKF = np.quaternion(1,0,0,0)
        self.quat2_EKF = np.quaternion(1,0,0,0)

    def callback1(self, data):
        self.quat1_imu = np.quaternion(data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z)

    def callback2(self, data):
        self.quat2_imu = np.quaternion(data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z)

    def callback1_EKF(self, data):
        self.quat1_EKF = np.quaternion(data.w, data.x, data.y, data.z)

    def callback2_EKF(self, data):
        self.quat2_EKF = np.quaternion(data.w, data.x, data.y, data.z)

    def main(self):
        rospy.init_node('angle_error', anonymous=True)
        rospy.Subscriber("/part1/Imu", Imu, self.callback1)
        rospy.Subscriber("/part2/Imu", Imu, self.callback2)
        rospy.Subscriber("/part1/EKF", Quaternion, self.callback1_EKF)
        rospy.Subscriber("/part2/EKF", Quaternion, self.callback2_EKF)
        
        pub = rospy.Publisher('/angle_error', Vector3, queue_size=10)
        rate = rospy.Rate(13.32)
        while not rospy.is_shutdown():
            imu_angle_vec = imu_angle(self.quat1_imu, self.quat2_imu)
            EKF_angle_vec = EKF_angle(self.quat1_EKF, self.quat2_EKF)
            angle_error = error_calc(imu_angle_vec, EKF_angle_vec)

            # print("quat1 imu: ", self.quat1_imu)
            # print("quat2 imu: ", self.quat2_imu)
            # print("imu angles: ", imu_angle_vec)
            # print("quat1 ekf: ", self.quat1_EKF)
            # print("quat2 ekf: ", self.quat2_EKF)
            # print("ekf angles: ", EKF_angle_vec)


            angle_error_pub = Vector3()
            angle_error_pub.x = angle_error[0]
            angle_error_pub.y = angle_error[1]
            angle_error_pub.z = angle_error[2]
            
            rospy.loginfo(angle_error_pub)
            pub.publish(angle_error_pub)
            rate.sleep()


if __name__ == '__main__':
    try:
         angle_error = AngleClass()
         angle_error.main()
    except rospy.ROSInterruptException:
        pass