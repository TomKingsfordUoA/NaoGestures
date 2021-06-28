#!/bin/env python2.7

import math

import pandas as pd

from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer


class NaoBvhConverter:
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

        return {
            "RShoulderRoll": bvh_rotations["RightArm_Zrotation"] - 90.0,
            "RShoulderPitch": -bvh_rotations["RightArm_Xrotation"] + 90.0,
            "RElbowYaw": -(bvh_rotations["RightForeArm_Zrotation"] + bvh_rotations["RightArm_Yrotation"]) + 90.0,
            "RElbowRoll": bvh_rotations["RightForeArm_Yrotation"],
            "LShoulderRoll": bvh_rotations["LeftArm_Zrotation"] + 90.0,
            "LShoulderPitch": -bvh_rotations["LeftArm_Xrotation"] + 90.0,
            "LElbowYaw": -(bvh_rotations["LeftForeArm_Zrotation"] + bvh_rotations["LeftArm_Yrotation"]) - 90.0,
            "LElbowRoll": bvh_rotations["LeftForeArm_Yrotation"],
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
