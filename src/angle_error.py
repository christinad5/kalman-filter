#!/usr/bin/env python3

import rospy
import numpy as np
import time
import quaternion
from std_msgs.msg import String, Float64, Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3

# calculate angles between Vectonav quaternion
# sorry for global variables, idk how to use the lambda functions yet :(

def callback1(data, quat1_imu_data):
    quat1_imu_data = np.quaternion(data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z)
    print('quat1_imu_data = ', quat1_imu_data)


def callback2(data, quat2_imu_data):
    quat2_imu_data = np.quaternion(data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z)
    print('quat2_imu_data = ', quat2_imu_data)


def imu_angle(quat1_imu, quat2_imu):
    q = quat1_imu * quat2_imu
    phi = np.arctan2((q.w*q.x + q.y*q.z), 1 - 2*(np.square(q.x)+np.square(q.y)))
    theta = np.arcsin(2*(q.w*q.y - q.z*q.x))
    psi = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(np.square(q.y)+np.square(q.z)))
    angles_rad = [phi, theta, psi]
    return angles_rad


def callback1_EKF(data, quat1_EKF_data):
    quat1_EKF_data = np.quaternion(data.w, data.x, data.y, data.z)


def callback2_EKF(data, quat2_EKF_data):
    quat2_EKF_data = np.quaternion(data.w, data.x, data.y, data.z)


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
        elif angles_imu == 0:
            if angles_EKF != 0:
                error = 100*np.abs(((angles_imu[i] - angles_EKF[i])/angles_EKF[i]))
                angle_error.append(error)
            elif angles_EKF ==0:
                error = 0
                angle_error.append(error)
        else:
            angle_error.append('8888')
    return angle_error


def listener():
    rospy.init_node('listener', anonymous=True)
    
    quat1_imu_data = np.quaternion(1,0,0,0)
    quat2_imu_data = np.quaternion(1,0,0,0)
    quat1_EKF_data = np.quaternion(1,0,0,0)
    quat2_EKF_data = np.quaternion(1,0,0,0)
    quat2_imu = lambda x: callback2(x, quat2_imu_data)
    quat1_EKF = lambda x: callback1_EKF(x, quat1_EKF_data)
    quat2_EKF = lambda x: callback2_EKF(x, quat2_EKF_data)
    

    rospy.Subscriber("/part1/Imu", Imu, lambda x: callback1(x, quat1_imu_data))
    rospy.Subscriber("/part2/Imu", Imu, quat2_imu)
    rospy.Subscriber("/part1/EKF", Quaternion, quat1_EKF)
    rospy.Subscriber("/part2/EKF", Quaternion, quat2_EKF)


    time.sleep(2) # this is so you give some time to subscriber
    pub = rospy.Publisher('/angle_error', Vector3, queue_size=10)
    rate = rospy.Rate(13.32)
    while not rospy.is_shutdown():

        imu_angle_vec = imu_angle(quat1_imu_data, quat2_imu_data)
        EKF_angle_vec = EKF_angle(quat1_EKF_data, quat2_EKF_data)
        angle_error = error_calc(imu_angle_vec, EKF_angle_vec)

        angle_error_pub = Vector3()
        angle_error_pub.x = angle_error[0]
        angle_error_pub.y = angle_error[1]
        angle_error_pub.z = angle_error[2]
        
        rospy.loginfo(angle_error_pub)
        pub.publish(angle_error_pub)
        rate.sleep()
    rospy.spin()


if __name__ == '__main__':
    listener()
