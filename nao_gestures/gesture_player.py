#!/bin/env python2.7

import sys
import time

import naoqi


class NaoGesturePlayer:
    def __init__(self, robot_ip, robot_port, my_ip, my_port, stream_err=sys.stderr):
        self._robot_ip = robot_ip
        self._robot_port = robot_port
        self._my_ip = my_ip
        self._my_port = my_port
        self._stream_err = stream_err

    def play(self, df_gestures, speed=1.0):
        """
        Takes a DataFrame of Nao joint rotations, with the index as a series of timestamps, and plays these on the Nao
        robot.
        """

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
