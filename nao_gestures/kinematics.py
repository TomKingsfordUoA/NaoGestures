import numpy as np
from scipy.spatial.transform import Rotation


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

        right_arm_length = np.linalg.norm(position_right_elbow_inertial - position_right_shoulder_standard)

        # The following is an analytical inversion of the forward kinematics:
        # p_elbow = R_y (-theta_p) R_z (theta_r) [-1 0 0]^T

        # There are two candidate rolls:
        theta_r = [
            normalize_angle(arcsin_safe(-p_elbow_hat[2])),
            normalize_angle(np.pi - arcsin_safe(-p_elbow_hat[2])),
        ]

        # There will be two candidates and we'll discard the one in the backward direction (most consistent with Nao's kinematics)
        theta_r = [angle for angle in theta_r if angle > -np.pi/2 and angle < np.pi/2]
        if len(theta_r) != 1:
            raise ValueError("Expected exactly one candidate theta_r")
        theta_r = theta_r[0]

        # Sanity check:
        if not np.isclose(np.sin(theta_r), -p_elbow_hat[2]):
            raise RuntimeError("Something broke")

        # In the case theta_r = -pi/2 => cos(theta_r) = 0, we have gimbal lock (and any pitch is admissible)
        if np.isclose(np.abs(theta_r), np.pi/2):
            theta_p = 0
            position_right_elbow_standard, rotation_right_elbow_standard = \
                ForwardKinematics.forward_kinematics_right_shoulder(
                    theta_right_shoulder_pitch=theta_p,
                    theta_right_shoulder_roll=theta_r,
                    position_right_shoulder_standard=position_right_shoulder_standard,
                    rotation_right_shoulder_standard=rotation_right_shoulder_standard,
                    right_arm_length=right_arm_length,
                )
            return theta_r, theta_p, position_right_elbow_standard, rotation_right_elbow_standard

        theta_p0 = [
            arccos_safe(-p_elbow_hat[1] / np.cos(theta_r)),
            -arccos_safe(-p_elbow_hat[1] / np.cos(theta_r)),
        ]
        theta_p1 = [
            arcsin_safe(-p_elbow_hat[0] / np.cos(theta_r)),
            np.pi - arcsin_safe(-p_elbow_hat[0] / np.cos(theta_r)),
        ]
        theta_p0 = [normalize_angle(angle) for angle in theta_p0]
        theta_p1 = [normalize_angle(angle) for angle in theta_p1]
        equivalence_matrix = [isclose_angles(a, b) for a in theta_p0 for b in theta_p1]
        if equivalence_matrix[0] or equivalence_matrix[1]:
            theta_p = theta_p0[0]
        elif equivalence_matrix[2] or equivalence_matrix[3]:
            theta_p = theta_p0[1]
        else:
            raise ValueError("Failed to find a pitch")

        position_right_elbow_standard, rotation_right_elbow_standard = \
            ForwardKinematics.forward_kinematics_right_shoulder(
                theta_right_shoulder_pitch=theta_p,
                theta_right_shoulder_roll=theta_r,
                position_right_shoulder_standard=position_right_shoulder_standard,
                rotation_right_shoulder_standard=rotation_right_shoulder_standard,
                right_arm_length=right_arm_length,
            )
        return theta_r, theta_p, position_right_elbow_standard, rotation_right_elbow_standard

    @staticmethod
    def inverse_kinematics_left_shoulder(
            position_left_shoulder_standard,
            rotation_left_shoulder_standard,
            position_left_elbow_inertial):

        # Convert right elbow position to right arm standard frame, relative to that frame's origin
        p_elbow_hat = make_unit(rotation_left_shoulder_standard.inv().apply(position_left_elbow_inertial - position_left_shoulder_standard))

        left_arm_length = np.linalg.norm(position_left_elbow_inertial - position_left_shoulder_standard)

        # The following is an analytical inversion of the forward kinematics:
        # p_elbow = R_y (-theta_p) R_z (theta_r) [-1 0 0]^T

        # There are two candidate rolls:
        theta_r = [
            normalize_angle(arcsin_safe(p_elbow_hat[2])),
            normalize_angle(np.pi - arcsin_safe(p_elbow_hat[2])),
        ]

        # There will be two candidates and we'll discard the one in the backward direction (most consistent with Nao's kinematics)
        theta_r = [angle for angle in theta_r if angle > -np.pi / 2 and angle < np.pi / 2]
        if len(theta_r) != 1:
            raise ValueError("Expected exactly one candidate theta_r")
        theta_r = theta_r[0]

        # Sanity check:
        if not np.isclose(np.sin(theta_r), p_elbow_hat[2]):
            raise RuntimeError("Something broke")

        # In the case theta_r = -pi/2 => cos(theta_r) = 0, we have gimbal lock (and any pitch is admissible)
        if np.isclose(np.abs(theta_r), np.pi / 2):
            theta_p = 0
            position_left_elbow_standard, rotation_left_elbow_standard = \
                ForwardKinematics.forward_kinematics_left_shoulder(
                    theta_left_shoulder_pitch=theta_p,
                    theta_left_shoulder_roll=theta_r,
                    position_left_shoulder_standard=position_left_shoulder_standard,
                    rotation_left_shoulder_standard=rotation_left_shoulder_standard,
                    left_arm_length=left_arm_length,
                )
            return theta_r, theta_p, position_left_elbow_standard, rotation_left_elbow_standard

        theta_p0 = [
            arccos_safe(p_elbow_hat[1] / np.cos(theta_r)),
            -arccos_safe(p_elbow_hat[1] / np.cos(theta_r)),
        ]
        theta_p1 = [
            arcsin_safe(-p_elbow_hat[0] / np.cos(theta_r)),
            np.pi - arcsin_safe(-p_elbow_hat[0] / np.cos(theta_r)),
        ]
        theta_p0 = [normalize_angle(angle) for angle in theta_p0]
        theta_p1 = [normalize_angle(angle) for angle in theta_p1]
        equivalence_matrix = [isclose_angles(a, b) for a in theta_p0 for b in theta_p1]
        if equivalence_matrix[0] or equivalence_matrix[1]:
            theta_p = theta_p0[0]
        elif equivalence_matrix[2] or equivalence_matrix[3]:
            theta_p = theta_p0[1]
        else:
            raise ValueError("Failed to find a pitch")

        position_left_elbow_standard, rotation_left_elbow_standard = \
            ForwardKinematics.forward_kinematics_left_shoulder(
                theta_left_shoulder_pitch=theta_p,
                theta_left_shoulder_roll=theta_r,
                position_left_shoulder_standard=position_left_shoulder_standard,
                rotation_left_shoulder_standard=rotation_left_shoulder_standard,
                left_arm_length=left_arm_length,
            )
        return theta_r, theta_p, position_left_elbow_standard, rotation_left_elbow_standard

    @staticmethod
    def inverse_kinematics_right_elbow(
            position_right_elbow_standard,
            rotation_right_elbow_standard,
            position_right_hand_inertial):

        # Calculate the position vector of the hand in the elbow standard frame:
        p_hat = make_unit(rotation_right_elbow_standard.inv().apply(position_right_hand_inertial - position_right_elbow_standard))

        right_forearm_length = np.linalg.norm(position_right_hand_inertial - position_right_elbow_standard)

        theta_r = [
            normalize_angle(arccos_safe(-p_hat[1])),
            normalize_angle(-arccos_safe(-p_hat[1])),
        ]

        # There will be two candidates and we'll discard the one in the backward direction (most consistent with Nao's kinematics)
        theta_r = {angle for angle in theta_r if angle >= 0}
        if len(theta_r) != 1:
            raise ValueError("Expected exactly one candidate theta_r")
        theta_r = next(theta_r.__iter__())

        # Detect gimbal lock:
        if np.isclose(theta_r, 0) or np.isclose(np.abs(theta_r), np.pi):
            theta_y = 0
            position_right_hand_standard, rotation_right_hand_standard = \
                ForwardKinematics.forward_kinematics_right_elbow(
                    theta_right_elbow_roll=theta_r,
                    theta_right_elbow_yaw=theta_y,
                    position_right_elbow_standard=position_right_elbow_standard,
                    rotation_right_elbow_standard=rotation_right_elbow_standard,
                    right_forearm_length=right_forearm_length,
                )
            return theta_r, theta_y, position_right_hand_standard, rotation_right_hand_standard

        theta_y0 = [
            arcsin_safe(p_hat[0] / np.sin(theta_r)),
            np.pi - arcsin_safe(p_hat[0] / np.sin(theta_r)),
        ]
        theta_y1 = [
            arccos_safe(-p_hat[2] / np.sin(theta_r)),
            -arccos_safe(-p_hat[2] / np.sin(theta_r)),
        ]
        equivalence_matrix = [isclose_angles(a, b) for a in theta_y0 for b in theta_y1]
        if equivalence_matrix[0] or equivalence_matrix[1]:
            theta_y = theta_y0[0]
        elif equivalence_matrix[2] or equivalence_matrix[3]:
            theta_y = theta_y0[1]
        else:
            raise ValueError("Failed to find a yaw")
        theta_y = normalize_angle(theta_y)

        position_right_hand_standard, rotation_right_hand_standard = \
            ForwardKinematics.forward_kinematics_right_elbow(
                theta_right_elbow_roll=theta_r,
                theta_right_elbow_yaw=theta_y,
                position_right_elbow_standard=position_right_elbow_standard,
                rotation_right_elbow_standard=rotation_right_elbow_standard,
                right_forearm_length=right_forearm_length,
            )
        return theta_r, theta_y, position_right_hand_standard, rotation_right_hand_standard

    @staticmethod
    def inverse_kinematics_left_elbow(
            position_left_elbow_standard,
            rotation_left_elbow_standard,
            position_left_hand_inertial):

        # Calculate the position vector of the hand in the elbow standard frame:
        p_hat = make_unit(
            rotation_left_elbow_standard.inv().apply(position_left_hand_inertial - position_left_elbow_standard))

        left_forearm_length = np.linalg.norm(position_left_hand_inertial - position_left_elbow_standard)

        theta_r = [
            normalize_angle(arccos_safe(p_hat[1])),
            normalize_angle(-arccos_safe(p_hat[1])),
        ]

        # There will be two candidates and we'll discard the one in the forward direction (most consistent with Nao's kinematics)
        theta_r = {angle for angle in theta_r if angle <= 0}
        if len(theta_r) != 1:
            raise ValueError("Expected exactly one candidate theta_r")
        theta_r = next(theta_r.__iter__())

        # Detect gimbal lock:
        if np.isclose(theta_r, 0) or np.isclose(np.abs(theta_r), np.pi):
            theta_y = 0
            position_left_hand_standard, rotation_left_hand_standard = \
                ForwardKinematics.forward_kinematics_left_elbow(
                    theta_left_elbow_roll=theta_r,
                    theta_left_elbow_yaw=theta_y,
                    position_left_elbow_standard=position_left_elbow_standard,
                    rotation_left_elbow_standard=rotation_left_elbow_standard,
                    left_forearm_length=left_forearm_length,
                )
            return theta_r, theta_y, position_left_hand_standard, rotation_left_hand_standard

        theta_y0 = [
            arcsin_safe(-p_hat[0] / np.sin(theta_r)),
            np.pi - arcsin_safe(-p_hat[0] / np.sin(theta_r)),
        ]
        theta_y1 = [
            arccos_safe(p_hat[2] / np.sin(theta_r)),
            -arccos_safe(p_hat[2] / np.sin(theta_r)),
        ]
        equivalence_matrix = [isclose_angles(a, b) for a in theta_y0 for b in theta_y1]
        if equivalence_matrix[0] or equivalence_matrix[1]:
            theta_y = theta_y0[0]
        elif equivalence_matrix[2] or equivalence_matrix[3]:
            theta_y = theta_y0[1]
        else:
            raise ValueError("Failed to find a yaw")
        theta_y = normalize_angle(theta_y)

        position_left_hand_standard, rotation_left_hand_standard = \
            ForwardKinematics.forward_kinematics_left_elbow(
                theta_left_elbow_roll=theta_r,
                theta_left_elbow_yaw=theta_y,
                position_left_elbow_standard=position_left_elbow_standard,
                rotation_left_elbow_standard=rotation_left_elbow_standard,
                left_forearm_length=left_forearm_length,
            )
        return theta_r, theta_y, position_left_hand_standard, rotation_left_hand_standard

    @staticmethod
    def inverse_kinematics(bvh_frames_plus_standard):
        position_right_shoulder_standard, rotation_right_shoulder_standard = bvh_frames_plus_standard['RightArmStandard']
        position_left_shoulder_standard, rotation_left_shoulder_standard = bvh_frames_plus_standard['LeftArmStandard']
        position_right_elbow_inertial, rotation_right_elbow_inertial = bvh_frames_plus_standard['RightForeArm']
        position_left_elbow_inertial, rotation_left_elbow_inertial = bvh_frames_plus_standard['LeftForeArm']
        position_right_hand_inertial, rotation_right_hand_inertial = bvh_frames_plus_standard['RightHand']
        position_left_hand_inertial, rotation_left_hand_inertial = bvh_frames_plus_standard['LeftHand']

        theta_right_shoulder_roll, theta_right_shoulder_pitch, position_right_elbow_standard, rotation_right_elbow_standard = \
            InverseKinematics.inverse_kinematics_right_shoulder(
                position_right_shoulder_standard=position_right_shoulder_standard,
                rotation_right_shoulder_standard=rotation_right_shoulder_standard,
                position_right_elbow_inertial=position_right_elbow_inertial,
            )
        theta_left_shoulder_roll, theta_left_shoulder_pitch, position_left_elbow_standard, rotation_left_elbow_standard = \
            InverseKinematics.inverse_kinematics_left_shoulder(
                position_left_shoulder_standard=position_left_shoulder_standard,
                rotation_left_shoulder_standard=rotation_left_shoulder_standard,
                position_left_elbow_inertial=position_left_elbow_inertial,
            )
        theta_right_elbow_roll, theta_right_elbow_yaw, position_right_hand_standard, rotation_right_hand_standard = \
            InverseKinematics.inverse_kinematics_right_elbow(
                position_right_elbow_standard=position_right_elbow_standard,
                rotation_right_elbow_standard=rotation_right_elbow_standard,
                position_right_hand_inertial=position_right_hand_inertial,
            )
        theta_left_elbow_roll, theta_left_elbow_yaw, position_left_hand_standard, rotation_left_hand_standard = \
            InverseKinematics.inverse_kinematics_left_elbow(
                position_left_elbow_standard=position_left_elbow_standard,
                rotation_left_elbow_standard=rotation_left_elbow_standard,
                position_left_hand_inertial=position_left_hand_inertial,
            )

        return {
            'RShoulderRoll': theta_right_shoulder_roll,
            'RShoulderPitch': theta_right_shoulder_pitch,
            'LShoulderRoll': theta_left_shoulder_roll,
            'LShoulderPitch': theta_left_shoulder_pitch,
            'RElbowRoll': theta_right_elbow_roll,
            'RElbowYaw': theta_right_elbow_yaw,
            'LElbowRoll': theta_left_elbow_roll,
            'LElbowYaw': theta_left_elbow_yaw,
        }


class ForwardKinematics:
    def __init__(self):
        raise NotImplementedError()

    @staticmethod
    def forward_kinematics_right_shoulder(
            theta_right_shoulder_pitch,
            theta_right_shoulder_roll,
            position_right_shoulder_standard,
            rotation_right_shoulder_standard,
            right_arm_length):

        rotation_right_elbow_standard = (
            rotation_right_shoulder_standard *
            Rotation.from_rotvec(theta_right_shoulder_pitch * np.array([0, 0, -1])) *
            Rotation.from_rotvec(theta_right_shoulder_roll * np.array([1, 0, 0]))
        )
        position_right_elbow_standard = \
            rotation_right_elbow_standard.apply(right_arm_length * np.array([0, -1, 0])) + position_right_shoulder_standard

        return position_right_elbow_standard, rotation_right_elbow_standard

    @staticmethod
    def forward_kinematics_left_shoulder(
            theta_left_shoulder_pitch,
            theta_left_shoulder_roll,
            position_left_shoulder_standard,
            rotation_left_shoulder_standard,
            left_arm_length):

        rotation_left_elbow_standard = (
                rotation_left_shoulder_standard *
                Rotation.from_rotvec(theta_left_shoulder_pitch * np.array([0, 0, 1])) *
                Rotation.from_rotvec(theta_left_shoulder_roll * np.array([1, 0, 0]))
        )
        position_left_elbow_standard = \
            rotation_left_elbow_standard.apply(left_arm_length * np.array([0, 1, 0])) + position_left_shoulder_standard

        return position_left_elbow_standard, rotation_left_elbow_standard

    @staticmethod
    def forward_kinematics_right_elbow(
            theta_right_elbow_roll,
            theta_right_elbow_yaw,
            position_right_elbow_standard,
            rotation_right_elbow_standard,
            right_forearm_length):

        rotation_right_hand_standard = (
            rotation_right_elbow_standard *
            Rotation.from_rotvec(theta_right_elbow_yaw * np.array([0, -1, 0])) *
            Rotation.from_rotvec(theta_right_elbow_roll * np.array([1, 0, 0]))
        )

        position_right_hand_standard = \
            rotation_right_hand_standard.apply(right_forearm_length * np.array([0, -1, 0])) + position_right_elbow_standard

        return position_right_hand_standard, rotation_right_hand_standard

    @staticmethod
    def forward_kinematics_left_elbow(
            theta_left_elbow_roll,
            theta_left_elbow_yaw,
            position_left_elbow_standard,
            rotation_left_elbow_standard,
            left_forearm_length):

        rotation_left_hand_standard = (
                rotation_left_elbow_standard *
                Rotation.from_rotvec(theta_left_elbow_yaw * np.array([0, -1, 0])) *
                Rotation.from_rotvec(theta_left_elbow_roll * np.array([1, 0, 0]))
        )

        position_left_hand_standard = \
            rotation_left_hand_standard.apply(
                left_forearm_length * np.array([0, 1, 0])) + position_left_elbow_standard

        return position_left_hand_standard, rotation_left_hand_standard

    @staticmethod
    def forward_kinematics(ik, bvh_frames_plus_standard):
        right_arm_length = np.linalg.norm(bvh_frames_plus_standard['RightArm'][0] - bvh_frames_plus_standard['RightForeArm'][0])
        left_arm_length = np.linalg.norm(bvh_frames_plus_standard['LeftArm'][0] - bvh_frames_plus_standard['LeftForeArm'][0])
        right_forearm_length = np.linalg.norm(bvh_frames_plus_standard['RightHand'][0] - bvh_frames_plus_standard['RightForeArm'][0])
        left_forearm_length = np.linalg.norm(bvh_frames_plus_standard['LeftHand'][0] - bvh_frames_plus_standard['LeftForeArm'][0])

        position_right_shoulder_standard, rotation_right_shoulder_standard = bvh_frames_plus_standard['RightArmStandard']
        position_left_shoulder_standard, rotation_left_shoulder_standard = bvh_frames_plus_standard['LeftArmStandard']

        position_right_elbow_standard, rotation_right_elbow_standard = ForwardKinematics.forward_kinematics_right_shoulder(
            ik['RShoulderPitch'],
            ik['RShoulderRoll'],
            position_right_shoulder_standard,
            rotation_right_shoulder_standard,
            right_arm_length)
        position_left_elbow_standard, rotation_left_elbow_standard = ForwardKinematics.forward_kinematics_left_shoulder(
            ik['LShoulderPitch'],
            ik['LShoulderRoll'],
            position_left_shoulder_standard,
            rotation_left_shoulder_standard,
            left_arm_length)
        position_right_hand_standard, rotation_right_hand_standard = ForwardKinematics.forward_kinematics_right_elbow(
            ik['RElbowRoll'],
            ik['RElbowYaw'],
            position_right_elbow_standard,
            rotation_right_elbow_standard,
            right_forearm_length)
        position_left_hand_standard, rotation_left_hand_standard = ForwardKinematics.forward_kinematics_left_elbow(
            ik['LElbowRoll'],
            ik['LElbowYaw'],
            position_left_elbow_standard,
            rotation_left_elbow_standard,
            left_forearm_length)

        return [
            ('FKRightElbow', position_right_elbow_standard, rotation_right_elbow_standard),
            ('FKLeftElbow', position_left_elbow_standard, rotation_left_elbow_standard),
            ('FKRightHand', position_right_hand_standard, rotation_right_hand_standard),
            ('FKLeftHand', position_left_hand_standard, rotation_left_hand_standard),
        ]


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


def isclose_angles(a, b, rtol=1e-3, atol=1e-3):
    if np.isclose(a, b, rtol=rtol, atol=atol):
        return True
    if np.isclose(2 * np.pi + a, b, rtol=rtol, atol=atol):
        return True
    if np.isclose(-2 * np.pi + a, b, rtol=rtol, atol=atol):
        return True
    return False
