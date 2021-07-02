#!/bin/env python2.7
import copy
import math

import numpy as np
import pandas as pd
from numpy import pi
from scipy.spatial.transform import Rotation

from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer


class NaoBvhConverter:
    """
    Reads a BVH motion capture file and converts into a dataframe of joint rotations which can be executed on a Nao
    robot.
    """

    def __init__(self):
        pass

    @staticmethod
    def read_mocap_data(bvh_file):
        parser = BVHParser()
        return parser.parse(bvh_file)

    @staticmethod
    def bvh_to_dataframe_of_nao_gestures(mocap_data):
        relevant_bvh_rotations = NaoBvhConverter._get_relevant_bvh_rotations(mocap_data)
        nao_rotations_degrees = NaoBvhConverter._convert_bvh_rotations_to_nao_degrees(relevant_bvh_rotations)
        nao_rotations_radians = {key: NaoBvhConverter._convert_series_degrees_to_radians(nao_rotations_degrees[key]) for
                                 key in nao_rotations_degrees}
        return NaoBvhConverter._convert_dict_of_series_to_df(nao_rotations_radians)

    @staticmethod
    def _get_relevant_bvh_rotations(mocap_data):
        """
        Get the subset of BVH rotations which will be used for robot gestures.
        """

        mp_euler = MocapParameterizer('euler')
        rotations, = mp_euler.fit_transform([mocap_data])
        relevant_frames = [
            "RightArm_Xrotation",
            "RightArm_Yrotation",
            "RightArm_Zrotation",
            "RightForeArm_Xrotation",
            "RightForeArm_Yrotation",
            "RightForeArm_Zrotation",
            "LeftArm_Xrotation",
            "LeftArm_Yrotation",
            "LeftArm_Zrotation",
            "LeftForeArm_Xrotation",
            "LeftForeArm_Yrotation",
            "LeftForeArm_Zrotation",
        ]
        return {key: rotations.values[key] for key in relevant_frames}

    @staticmethod
    def _convert_bvh_rotations_to_nao_degrees(bvh_rotations):
        """
        Take an input dictionary of series of BVH rotations (in degrees), and convert to Nao frames (in degrees)
        """

        theta = 5 * math.pi / 4

        return {
            "RShoulderRoll": bvh_rotations["RightArm_Zrotation"] - 90.0,
            "RShoulderPitch": -bvh_rotations["RightArm_Xrotation"] + 90.0,
            "RElbowYaw": (math.cos(theta) * bvh_rotations["RightForeArm_Zrotation"] + math.sin(theta) * bvh_rotations["RightArm_Yrotation"]) + 90.0,
            "RElbowRoll": bvh_rotations["RightForeArm_Yrotation"],
            "RWristYaw": bvh_rotations["RightForeArm_Xrotation"],
            "LShoulderRoll": bvh_rotations["LeftArm_Zrotation"] + 90.0,
            "LShoulderPitch": -bvh_rotations["LeftArm_Xrotation"] + 90.0,
            "LElbowYaw": (math.cos(theta) * bvh_rotations["LeftForeArm_Zrotation"] + math.sin(theta) * bvh_rotations["LeftArm_Yrotation"]) - 90.0,
            "LElbowRoll": bvh_rotations["LeftForeArm_Yrotation"],
            "LWristYaw": bvh_rotations["LeftForeArm_Xrotation"],
        }

    @staticmethod
    def _convert_series_degrees_to_radians(series_degrees):
        """
        Converts a series of floating point numbers in degrees to radians.
        """

        return series_degrees * math.pi / 180.0

    @staticmethod
    def _convert_dict_of_series_to_df(dict_of_series):
        return pd.DataFrame(data=dict_of_series)

    @staticmethod
    def get_bvh_frames(mocap_data):
        skeleton = mocap_data.skeleton

        # Get index:
        index = mocap_data.values['Hips_Xrotation'].index
        # TODO(TK): check to ensure all indices are the same

        all_frames = []
        n_frames = mocap_data.values['Hips_Xrotation'].size
        for idx_t in range(n_frames):

            # Initialize the hips arbitrarily (as this is our reference):
            frames = {'Hips': (np.array([0, 0, 80]), Rotation.from_euler('zxy', [0, pi/2, 0]))}

            # Breadth first search over tree:
            frontier = {child for child in skeleton['Hips']['children']}
            while len(frontier) != 0:
                frame_name = frontier.pop()
                frame = skeleton[frame_name]
                parent_name = frame['parent']
                if len(frame['channels']) == 0:
                    continue

                # Pose in parent's frame:
                position_child = np.array(frame['offsets'])  # xyz
                rotation_x = mocap_data.values[frame_name + '_Xrotation'].iloc[idx_t]
                rotation_y = mocap_data.values[frame_name + '_Yrotation'].iloc[idx_t]
                rotation_z = mocap_data.values[frame_name + '_Zrotation'].iloc[idx_t]
                rotation_child = Rotation.from_euler('zxy', [rotation_z, rotation_x, rotation_y], degrees=True)

                # Parent's pose in Hips' frame:
                position_parent, rotation_parent = frames[parent_name]

                # Calculate child's pose in Hips' frame:
                rotation = rotation_parent * rotation_child  # we want to R_0 R_1 ... R_n so child then parent
                position = rotation_parent.apply(position_child) + position_parent  # offset is in parent's frame

                # Add to tree:
                frames[frame_name] = (position, rotation)

                frontier = frontier.union(frame['children'])

            all_frames.append(frames)

        return all_frames, index

    @staticmethod
    def add_standard_frames(bvh_frames):
        """
        The goal here is to create some shoulder-attached frames which reference anatomical landmarks. The BVH frames are
        arbitrary and make it difficult to perform inverse kinematics and solve for Nao robot joint rotations. With
        anatomically-referenced frames, this becomes easier.
        """

        # Grab the references:
        position_right_arm, rotation_right_arm = bvh_frames['RightArm']
        position_left_arm, rotation_left_arm = bvh_frames['LeftArm']
        position_hips, rotation_hips = bvh_frames['Hips']

        # Calculate the normal to the shoulder/hip plane
        # The normal vector is orthogonal to the vector between the shoulders and the vector from a shoulder to the hips
        # this is a system of linear equations expressing this. The last row adds an arbitrary constraint that the sum of
        # the components is one so the system is full rank and uniquely solvable.
        A = np.array([
            position_left_arm - position_right_arm,
            position_left_arm - position_hips,
            [np.random.random(), np.random.random(), np.random.random()]]
        )
        b = np.array([[0], [0], [1]])
        n = np.linalg.solve(A, b)
        n_hat = n / np.linalg.norm(n)  # make it a unit vector
        n_hat = n_hat.reshape([-1])
        # Make it point in the forward direction for the robot
        if n_hat[1] < 0:
            n_hat *= -1
        # TODO(TK): sanity check that n_hat is orthogonal to the two vectors

        # We wish to take the right arm frame and rotate it such that the y axis is parallel with n_hat
        y_r = rotation_right_arm.apply([0, 1, 0])
        theta_r = np.arccos(np.dot(y_r, n_hat))  # noting that each vector is of unit length already
        rot_vec_r = np.cross(y_r, n_hat)  # this is a unit vector orthogonal to y_r and n_hat, oriented by the right hand rule
        rotation_right_shoulder_standard = Rotation.from_rotvec(theta_r * rot_vec_r) * rotation_right_arm

        # We wish to take the right arm standard frame and rotate it about its y axis such that the z axis is parallel with
        # the vector between the left and right arm
        z_r_standard = rotation_right_shoulder_standard.apply([0, 0, 1])
        theta_r = -np.arccos(np.dot((position_right_arm - position_left_arm)/np.linalg.norm(position_right_arm - position_left_arm), z_r_standard))
        y_r_standard = rotation_right_shoulder_standard.apply([0, 1, 0])
        rotation_right_shoulder_standard = Rotation.from_rotvec(theta_r * y_r_standard) * rotation_right_shoulder_standard

        # We wish to take the left arm frame and rotate it such that the y axis is parallel with n_hat, in the opposite direction
        y_l = rotation_left_arm.apply([0, 1, 0])
        theta_l = -np.arccos(np.dot(y_l, -n_hat))  # noting that each vector is of unit length already
        rot_vec_l = np.cross(y_l, n_hat)  # this is a unit vector orthogonal to y_l and n_hat, oriented by the right hand rule
        rotation_left_arm_standard = Rotation.from_rotvec(theta_l * rot_vec_l) * rotation_left_arm

        # We wish to take the left arm standard frame and rotate it about its y axis such that the z axis is parallel with
        # the vector between the left and right arm
        z_l_standard = rotation_left_arm_standard.apply([0, 0, 1])
        theta_l = np.arccos(np.dot((position_left_arm - position_right_arm) / np.linalg.norm(position_left_arm - position_right_arm), z_l_standard))
        y_l_standard = rotation_left_arm_standard.apply([0, 1, 0])
        rotation_left_arm_standard = Rotation.from_rotvec(theta_l * y_l_standard) * rotation_left_arm_standard

        # Copy:
        bvh_frames_plus_standard = copy.deepcopy(bvh_frames)
        bvh_frames_plus_standard['RightArmStandard'] = (position_right_arm, rotation_right_shoulder_standard)
        bvh_frames_plus_standard['LeftArmStandard'] = (position_left_arm, rotation_left_arm_standard)

        return bvh_frames_plus_standard