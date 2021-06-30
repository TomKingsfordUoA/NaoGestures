import copy
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa
from numpy import pi
from scipy.spatial.transform import Rotation

from nao_gestures import NaoGesturePlayer
from pymo.parsers import BVHParser


def draw_frame(ax, label, position, rotation, arrow_length=1.0):
    x = rotation.apply(np.array([1, 0, 0]))
    y = rotation.apply(np.array([0, 1, 0]))
    z = rotation.apply(np.array([0, 0, 1]))

    colors = ['r', 'g', 'b']
    for axis, color in zip([x, y, z], colors):
        ax.quiver(
            [position[0]],
            [position[1]],
            [position[2]],
            [axis[0]],
            [axis[1]],
            [axis[2]],
            color=color,
            length=arrow_length,
            linewidths=1,
        )
        label_position = position - 0.2 * arrow_length * (x + y + z) / np.sqrt(3)
        ax.text(label_position[0], label_position[1], label_position[2], label, style='italic')


def get_bvh_frames(bvh_file):
    parser = BVHParser()
    mocap_data = parser.parse(bvh_file)
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


def make_unit(vector):
    n = np.linalg.norm(vector)
    if np.isclose(n, 0.0):
        return vector
    else:
        return vector / n


def normalize_angle(angle_radians):
    while angle_radians > np.pi:
        angle_radians -= 2 * np.pi
    while angle_radians <= -np.pi:
        angle_radians += 2 * np.pi
    return angle_radians


def angle_between(a, b, reference_direction):
    a_hat = make_unit(a)
    b_hat = make_unit(b)

    theta = normalize_angle(np.arccos(np.dot(a_hat, b_hat)))

    # If theta is 0 or pi, we have colinear vectors and needn't consider the reference direction:
    if np.isclose(theta, 0) or np.isclose(theta, np.pi):
        return theta

    # Find a rotation vector which is in the reference direction:
    c = make_unit(np.cross(a_hat, b_hat))
    if np.arccos(np.dot(c, reference_direction)) > np.pi / 2:
        c *= -1
    rot = Rotation.from_rotvec(c * theta)

    if np.allclose(rot.apply(a_hat), b_hat):
        return theta
    elif np.allclose(rot.inv().apply(a_hat), b_hat):
        return -theta
    else:
        raise ValueError()


def arccos_safe(arg):
    if np.isclose(arg, 1):
        arg = 1
    if np.isclose(arg, -1):
        arg = -1
    return np.arccos(arg)


def arcsin_safe(arg):
    if np.isclose(arg, 1):
        arg = 1
    if np.isclose(arg, -1):
        arg = -1
    return np.arcsin(arg)


class InverseKinematics:
    def __init__(self):
        raise NotImplementedError()

    @staticmethod
    def inverse_kinematics_right_shoulder(
            position_right_shoulder_standard,
            rotation_right_shoulder_standard,
            position_right_elbow_inertial):

        # Convert right elbow position to right arm standard frame, relative to that frame's origin
        p_elbow_hat = make_unit(rotation_right_shoulder_standard.inv().apply(position_right_elbow_inertial - position_right_shoulder_standard))

        # The following is an analytical inversion of the forward kinematics:
        # p_elbow = R_y (-theta_p) R_z (theta_r) [-1 0 0]^T

        # There are two candidate rolls:
        theta_r = [
            normalize_angle(arcsin_safe(-p_elbow_hat[1])),
            normalize_angle(np.pi - arcsin_safe(-p_elbow_hat[1])),
        ]

        # There will be two candidates and we'll discard the one in the backward direction (most consistent with Nao's kinematics)
        theta_r = [angle for angle in theta_r if angle > -np.pi/2 and angle < np.pi/2]
        if len(theta_r) != 1:
            raise ValueError("Expected exactly one candidate theta_r")
        theta_r = theta_r[0]

        # Sanity check:
        if not np.isclose(np.sin(theta_r), -p_elbow_hat[1]):
            raise RuntimeError("Something broke")

        # In the case theta_r = -pi/2 => cos(theta_r) = 0, we have gimbal lock (and any pitch is admissible)
        if np.isclose(np.abs(theta_r), np.pi/2):
            return {
                'RShoulderRoll': theta_r,
                'RShoulderPitch': 0,
            }

        theta_p0 = [
            arccos_safe(-p_elbow_hat[0] / np.cos(theta_r)),
            -arccos_safe(-p_elbow_hat[0] / np.cos(theta_r)),
        ]
        theta_p1 = [
            arcsin_safe(-p_elbow_hat[2] / np.cos(theta_r)),
            np.pi - arcsin_safe(-p_elbow_hat[2] / np.cos(theta_r)),
        ]
        theta_p0 = [normalize_angle(angle) for angle in theta_p0]
        theta_p1 = [normalize_angle(angle) for angle in theta_p1]
        equivalence_matrix = [np.isclose(a, b) for a in theta_p0 for b in theta_p1]
        if equivalence_matrix[0] or equivalence_matrix[1]:
            theta_p = theta_p0[0]
        elif equivalence_matrix[2] or equivalence_matrix[3]:
            theta_p = theta_p0[1]
        else:
            raise ValueError("Failed to find a pitch")

        return {
            'RShoulderRoll': theta_r,
            'RShoulderPitch': theta_p,
        }

    @staticmethod
    def inverse_kinematics(bvh_frames_plus_standard):
        position_right_shoulder_standard, rotation_right_shoulder_standard = bvh_frames_plus_standard['RightArmStandard']
        position_right_elbow_inertial, rotation_right_elbow_inertial = bvh_frames_plus_standard['RightForeArm']

        ik = dict()
        ik.update(InverseKinematics.inverse_kinematics_right_shoulder(
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            position_right_elbow_inertial=position_right_elbow_inertial,
        ))
        return ik


class ForwardKinematics:
    def __init__(self):
        raise NotImplementedError()

    @staticmethod
    def forward_kinematics_right_elbow(
            theta_right_shoulder_pitch,
            theta_right_shoulder_roll,
            position_right_shoulder_standard,
            rotation_right_shoulder_standard,
            right_arm_length):

        rotation_right_elbow_standard = (
            Rotation.from_rotvec(theta_right_shoulder_pitch * np.array([0, -1, 0])) *
            Rotation.from_rotvec(theta_right_shoulder_roll * np.array([0, 0, 1])) *
            rotation_right_shoulder_standard
        )
        position_right_elbow_standard = \
            rotation_right_elbow_standard.apply(right_arm_length * np.array([-1, 0, 0])) + position_right_shoulder_standard

        return position_right_elbow_standard, rotation_right_elbow_standard

    @staticmethod
    def forward_kinematics(ik, bvh_frames_plus_standard):
        right_arm_length = np.linalg.norm(bvh_frames_plus_standard['RightArm'][0] - bvh_frames_plus_standard['RightForeArm'][0])

        position_right_shoulder_standard, rotation_right_shoulder_standard = bvh_frames_plus_standard['RightArmStandard']

        theta_right_shoulder_pitch = ik['RShoulderPitch']
        theta_right_shoulder_roll = ik['RShoulderRoll']

        position_right_elbow_standard, rotation_right_elbow_standard = ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch,
            theta_right_shoulder_roll,
            position_right_shoulder_standard,
            rotation_right_shoulder_standard,
            right_arm_length)

        return [
            ('FKRightElbow', position_right_elbow_standard, rotation_right_elbow_standard),
        ]


def main():
    all_frames, index = get_bvh_frames('nao_gestures/demos/examples/rand_5.bvh')
    all_frames = [add_standard_frames(frames) for frames in all_frames]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    drawable_frames = [
        # 'Hips',
        # 'LeftShoulder',
        # 'RightShoulder',
        # 'LeftArm',
        # 'RightArm',
        'LeftArmStandard',
        'RightArmStandard',
        # 'LeftForeArm',
        'RightForeArm',
        # 'LeftHand',
        'RightHand',
        # 'LeftUpLeg',
        # 'RightUpLeg',
        # 'LeftLeg',
        # 'RightLeg',
        # 'LeftFoot',
        # 'RightFoot',
        # 'Spine',
        # 'Spine1',
        # 'Spine2',
        # 'Spine3',
        'Head',
    ]
    ik = []
    for idx, frames in enumerate(all_frames):
        ax.cla()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')  # noqa
        ax.set_title(str(0.05 * idx))
        reference_dim = 150
        ax.set_xlim([-reference_dim / 2, reference_dim / 2])  # noqa
        ax.set_ylim([-reference_dim / 2, reference_dim / 2])  # noqa
        ax.set_zlim([0, reference_dim])  # noqa

        ik.append(InverseKinematics.inverse_kinematics(frames))
        # ik.append(inverse_kinematics(frames))
        # print(ik[-1]['RElbowRoll'] * 180 / pi, ik[-1]['RElbowYaw'] * 180 / pi)

        for frame_name, position, rotation in ForwardKinematics.forward_kinematics(ik[-1], frames):
            draw_frame(ax, frame_name, position, rotation, arrow_length=10)

        for frame_name, (position, rotation) in frames.items():
            if len(drawable_frames) == 0 or frame_name in drawable_frames:
                draw_frame(ax, frame_name, position, rotation, arrow_length=10)
        plt.pause(0.05)

    # Play on Nao:
    player = NaoGesturePlayer(
        robot_ip="127.0.0.1",
        robot_port=9559,
        my_ip="127.0.0.1",
        my_port=1234,
    )
    df_gestures = pd.DataFrame(data=ik, index=index)
    # player.play(df_gestures, speed=1.0)


if __name__ == '__main__':
    main()
