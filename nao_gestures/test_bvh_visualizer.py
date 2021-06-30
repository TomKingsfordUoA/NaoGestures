import numpy as np
from scipy.spatial.transform import Rotation

from nao_gestures.bvh_visualizer import ForwardKinematics, InverseKinematics


def test_forward_kinematics_right_elbow_zero():
    # Zero control input:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=0,
            theta_right_shoulder_roll=0,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    assert np.allclose(position_right_elbow_standard, np.array([-1, 0, 0]))
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       Rotation.from_euler('zxy', [0, 0, 0]).as_rotvec())


def test_forward_kinematics_right_elbow_shoulder_roll_only():
    # Shoulder roll only:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=0,
            theta_right_shoulder_roll=-np.pi/2,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    assert np.allclose(position_right_elbow_standard, np.array([0, 1, 0]))
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       Rotation.from_euler('zxy', [-np.pi/2, 0, 0]).as_rotvec())


def test_forward_kinematics_right_elbow_shoulder_pitch_only():
    # Shoulder pitch only:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=np.pi/2,
            theta_right_shoulder_roll=0,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    assert np.allclose(position_right_elbow_standard, np.array([0, 0, -1]))
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       Rotation.from_euler('zxy', [0, 0, -np.pi/2]).as_rotvec())


def test_forward_kinematics_right_elbow_shoulder_roll_and_pitch():
    # Shoulder roll and pitch:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=np.pi / 4,
            theta_right_shoulder_roll=-np.pi / 4,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )
    assert np.allclose(position_right_elbow_standard, np.array([-0.5, 1/np.sqrt(2), -0.5]))
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       Rotation.from_euler('xzy', [0, -np.pi / 4, -1 * np.pi / 4]).as_rotvec())  # pitch (-y axis) then roll (+z axis)


def test_inverse_kinematics_right_shoulder_zero():
    # Zero control input:
    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=np.array([-1, 0, 0]),
    )
    assert np.allclose(ik['RShoulderRoll'], 0)
    assert np.allclose(ik['RShoulderPitch'], 0)


def test_inverse_kinematics_right_shoulder_shoulder_roll_only():
    # Shoulder roll only:
    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0]),
    )
    assert np.allclose(ik['RShoulderRoll'], -np.pi/4)
    assert np.allclose(ik['RShoulderPitch'], 0)


def test_inverse_kinematics_right_shoulder_shoulder_pitch_only():
    # Shoulder pitch only:
    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=np.array([0, 0, -1]),
    )
    assert np.allclose(ik['RShoulderRoll'], 0)
    assert np.allclose(ik['RShoulderPitch'], np.pi/2)


def test_inverse_kinematics_right_shoulder_shoulder_roll_and_pitch():
    # Shoulder roll and pitch:
    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=np.array([-1.0/2, 1/np.sqrt(2), -1.0/2]),
    )
    assert np.allclose(ik['RShoulderRoll'], -np.pi / 4)
    assert np.allclose(ik['RShoulderPitch'], np.pi / 4)


def test_forward_then_inverse_kinematics_random_pitch():
    np.random.seed(43)
    theta_right_shoulder_pitch = np.random.random() * 2 * np.pi - np.pi
    theta_right_shoulder_roll = 0

    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=theta_right_shoulder_pitch,
            theta_right_shoulder_roll=theta_right_shoulder_roll,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=position_right_elbow_standard,
    )

    assert np.isclose(ik['RShoulderPitch'], theta_right_shoulder_pitch)
    assert np.isclose(ik['RShoulderRoll'], theta_right_shoulder_roll)


def test_forward_then_inverse_kinematics_random_roll():
    np.random.seed(42)
    theta_right_shoulder_pitch = 0
    theta_right_shoulder_roll = np.random.random() * np.pi - np.pi / 2

    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=theta_right_shoulder_pitch,
            theta_right_shoulder_roll=theta_right_shoulder_roll,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=position_right_elbow_standard,
    )

    assert np.isclose(ik['RShoulderPitch'], theta_right_shoulder_pitch)
    assert np.isclose(ik['RShoulderRoll'], theta_right_shoulder_roll)


def test_forward_then_inverse_kinematics_random_pitch_and_roll():
    np.random.seed(12)
    theta_right_shoulder_pitch = np.random.random() * 2 * np.pi - np.pi
    theta_right_shoulder_roll = np.random.random() * np.pi - np.pi / 2

    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=theta_right_shoulder_pitch,
            theta_right_shoulder_roll=theta_right_shoulder_roll,
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    # we know that position_right_elbow_standard[1] = -sin(theta_r)
    assert np.isclose(position_right_elbow_standard[1], -np.sin(theta_right_shoulder_roll))

    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=position_right_elbow_standard,
    )

    assert np.isclose(ik['RShoulderPitch'], theta_right_shoulder_pitch)
    assert np.isclose(ik['RShoulderRoll'], theta_right_shoulder_roll)


def test_inverse_then_forward_kinematics_random():
    np.random.seed(12)
    position_right_elbow_standard_initial = np.random.random([3])

    ik = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=np.array([0, 0, 0]),
        rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
        position_right_elbow_inertial=position_right_elbow_standard_initial,
    )

    # we know that position_right_elbow_standard[1] = -sin(theta_r) * np.linalg.norm(position_right_elbow_standard)
    assert np.isclose(-np.sin(ik['RShoulderRoll']) * np.linalg.norm(position_right_elbow_standard_initial), position_right_elbow_standard_initial[1])

    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=ik['RShoulderPitch'],
            theta_right_shoulder_roll=ik['RShoulderRoll'],
            position_right_shoulder_standard=np.array([0, 0, 0]),
            rotation_right_shoulder_standard=Rotation.from_euler('zxy', [0, 0, 0]),
            right_arm_length=1.0,
        )

    # Un-normalize length:
    position_right_elbow_standard *= np.linalg.norm(position_right_elbow_standard_initial)

    assert np.allclose(position_right_elbow_standard, position_right_elbow_standard_initial)
