#!/bin/env python2.7
import os
import socket
import sys
import time

import pandas as pd
import pymo.data

from nao_gestures import NaoBvhConverter
from nao_gestures.mocap_data_helpers import mocap_data_from_json


class NaoGesturePlayer:
    """
    Plays a dataframe of Nao joint rotations on a Nao robot.
    """

    def __init__(self):
        pass

    def play(self, df_gestures, speed=1.0):
        """
        Takes a DataFrame of Nao joint rotations, with the index as a series of timestamps, and plays these on the Nao
        robot.

        Subclasses should override this method.
        """
        raise NotImplementedError()


class NaoqiNaoGesturePlayer(NaoGesturePlayer):
    def __init__(self, robot_ip, robot_port, my_ip, my_port, stream_err=sys.stderr):
        NaoGesturePlayer.__init__(self)
        self._robot_ip = robot_ip
        self._robot_port = robot_port
        self._my_ip = my_ip
        self._my_port = my_port
        self._stream_err = stream_err

    def play(self, df_gestures, speed=1.0):
        # Perform import at runtime, not execute time as naoqi isn't a hard requirement
        # try/except not really necessary, just makes it clear we're very much doing a runtime import
        try:
            import naoqi
        except ImportError as exc:
            raise exc

        proxy_motion = naoqi.ALProxy("ALMotion", self._robot_ip, self._robot_port)
        proxy_motion.wakeUp()

        dt_initial = 0.1
        t_begin = time.time() + dt_initial
        for index, series_row in df_gestures.iterrows():
            # Sync timing:
            t_target = index.total_seconds()
            t_target /= speed
            t_elapsed = time.time() - t_begin
            dt = t_target - t_elapsed
            if dt > 0:
                time.sleep(dt)
            else:
                self._stream_err.write("WARNING! Robot fell behind achieving gestures\n")

            # Execute on robot:
            for frame in series_row.index:
                proxy_motion.setAngles(frame, series_row[frame], 1.0)


class RosNaoGesturePlayer(NaoGesturePlayer):
    """
    ros-related imports are performed at runtime, not import time as rospy isn't a hard requirement
    rospy is placed on PYTHONPATH by `source /opt/ros/melodic/setup.bash`
    """
    def __init__(self, ros_master_uri, my_ip_or_hostname, topic_name='gestures_bvh'):
        NaoGesturePlayer.__init__(self)
        self._ros_master_uri = ros_master_uri
        self._topic_name = topic_name

        try:
            socket.inet_aton(my_ip_or_hostname)
            # if abvoe doesn't error, this is a valid ip address
            self._ros_ip = my_ip_or_hostname
            self._ros_hostname = None
        except socket.error:
            # if not an ip, treat it as a hostname
            self._ros_ip = None
            self._ros_hostname = my_ip_or_hostname

    # staticmethod
    def __manage_environment(func):
        def _func(self, *args, **kwargs):
            targets = {
                'ROS_MASTER_URI': self._ros_master_uri,
                'ROS_IP': self._ros_ip,
                'ROS_HOSTNAME': self._ros_hostname,
            }

            targets_before = {
                key: os.environ[key] if key in os.environ else None
                for key in targets
            }

            # Set target environment variables:
            for key, value in targets.items():
                if value is not None:
                    os.environ[key] = value
                else:
                    if key in os.environ:
                        del os.environ[key]

            # Execute wrapped function:
            func(self, *args, **kwargs)

            # Recover initial environment variables:
            for key, value in targets_before.items():
                if value is not None:
                    os.environ[key] = value
                else:
                    if key in os.environ:
                        del os.environ[key]

        return _func

    @__manage_environment
    def play(self, df_gestures, speed=1.0):
        import rospy
        from naoqi_bridge_msgs.msg import JointAnglesWithSpeed

        if 'ROS_MASTER_URI' not in os.environ:
            raise ValueError('Environment variable ROS_MASTER_URI not set')
        if 'ROS_IP' not in os.environ and 'ROS_HOSTNAME' not in os.environ:
            raise ValueError('Neither environment variable ROS_IP nor ROS_HOSTNAME is set')

        pub = rospy.Publisher('joint_angles', JointAnglesWithSpeed, queue_size=10)
        rospy.init_node('nao-gestures', anonymous=True)

        dt_initial = 0.1
        t_begin = time.time() + dt_initial
        for index, series_row in df_gestures.iterrows():
            if rospy.is_shutdown():
                return

            # Sync timing:
            t_target = index.total_seconds()
            t_target /= speed
            t_elapsed = time.time() - t_begin
            dt = t_target - t_elapsed
            if dt > 0:
                rospy.sleep(dt)
            else:
                rospy.logwarn("WARNING! Robot fell behind achieving gestures\n")

            # Execute on robot:
            joint_angles_with_speed = JointAnglesWithSpeed()
            for frame in series_row.index:
                joint_angles_with_speed.joint_names.append(frame)
                joint_angles_with_speed.joint_angles.append(series_row[frame])
            joint_angles_with_speed.speed = 1.0
            rospy.loginfo(joint_angles_with_speed)
            pub.publish(joint_angles_with_speed)

    def _mocap_callback(self, data):
        """
        pymo's MocapData is pickled and published to the topic as a String, rather than using for instance https://wiki.ros.org/bvh_broadcaster
        because that library publishes tf transforms, and this is undesirable because MocapData is just an intermediate format
        used by co-speech gesture models and is based on arbitrary human kinematics rather than physically-meaningful robot
        kinematics.
        """
        import rospy

        rospy.loginfo("Received mocap data")
        sys.stdout.flush()

        mocap_data = mocap_data_from_json(data.data)

        df_gestures = NaoBvhConverter.bvh_to_dataframe_of_nao_gestures(mocap_data)
        self.play(df_gestures)

    def run(self):
        import rospy
        from std_msgs.msg import String

        rospy.init_node('nao-gestures', anonymous=True)
        rospy.Subscriber(self._topic_name, String, self._mocap_callback)
        rospy.spin()
