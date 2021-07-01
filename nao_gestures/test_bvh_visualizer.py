import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from nao_gestures.bvh_visualizer import ForwardKinematics, InverseKinematics


def test_right_shoulder_forward_kinematics_zero():
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Zero control input:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=0,
            theta_right_shoulder_roll=0,
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            right_arm_length=1.0,
        )

    position_right_elbow_standard_expected = rotation_right_shoulder_standard.apply(np.array([0, -1, 0])) + position_right_shoulder_standard

    assert np.allclose(position_right_elbow_standard, position_right_elbow_standard_expected)
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       rotation_right_shoulder_standard.as_rotvec())


def test_right_shoulder_forward_kinematics_shoulder_roll_only():
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Shoulder roll only:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=0,
            theta_right_shoulder_roll=-np.pi/2,
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            right_arm_length=1.0,
        )

    position_right_elbow_standard_expected = rotation_right_shoulder_standard.apply(np.array([0, 0, 1])) + position_right_shoulder_standard

    assert np.allclose(position_right_elbow_standard, position_right_elbow_standard_expected)
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       (rotation_right_shoulder_standard * Rotation.from_euler('zxy', [0, -np.pi/2, 0])).as_rotvec())


def test_right_shoulder_forward_kinematics_shoulder_pitch_only():
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Shoulder pitch only:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=np.pi/2,
            theta_right_shoulder_roll=0,
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            right_arm_length=1.0,
        )

    position_right_elbow_standard_expected = rotation_right_shoulder_standard.apply(np.array([-1, 0, 0])) + position_right_shoulder_standard

    assert np.allclose(position_right_elbow_standard, position_right_elbow_standard_expected)
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       (rotation_right_shoulder_standard * Rotation.from_euler('zxy', [-np.pi/2, 0, 0])).as_rotvec())


def test_right_shoulder_forward_kinematics_shoulder_roll_and_pitch():
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Shoulder roll and pitch:
    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=np.pi / 4,
            theta_right_shoulder_roll=-np.pi / 4,
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            right_arm_length=1.0,
        )

    position_right_elbow_standard_expected = rotation_right_shoulder_standard.apply(np.array([-0.5, -0.5, 1/np.sqrt(2)])) + position_right_shoulder_standard

    assert np.allclose(position_right_elbow_standard, position_right_elbow_standard_expected)
    assert np.allclose(rotation_right_elbow_standard.as_rotvec(),
                       (rotation_right_shoulder_standard * Rotation.from_euler('xzy', [-np.pi / 4, -1 * np.pi / 4, 0])).as_rotvec())  # pitch (-y axis) then roll (+z axis)


def test_right_shoulder_inverse_kinematics_zero():
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Zero control input:
    theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=position_right_shoulder_standard,
        rotation_right_shoulder_standard=rotation_right_shoulder_standard,
        position_right_elbow_inertial=rotation_right_shoulder_standard.apply(np.array([0, -1, 0])) + position_right_shoulder_standard,
    )
    assert np.allclose(theta_r, 0)
    assert np.allclose(theta_p, 0)


def test_right_shoulder_inverse_kinematics_shoulder_roll_only():    
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Shoulder roll only:
    theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=position_right_shoulder_standard,
        rotation_right_shoulder_standard=rotation_right_shoulder_standard,
        position_right_elbow_inertial=rotation_right_shoulder_standard.apply(np.array([0, -1/np.sqrt(2), 1/np.sqrt(2)])) + position_right_shoulder_standard,
    )
    assert np.allclose(theta_r, -np.pi/4)
    assert np.allclose(theta_p, 0)


def test_right_shoulder_inverse_kinematics_shoulder_pitch_only():
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Shoulder pitch only:
    theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=position_right_shoulder_standard,
        rotation_right_shoulder_standard=rotation_right_shoulder_standard,
        position_right_elbow_inertial=rotation_right_shoulder_standard.apply(np.array([-1, 0, 0])) + position_right_shoulder_standard,
    )
    assert np.allclose(theta_r, 0)
    assert np.allclose(theta_p, np.pi/2)


def test_right_shoulder_inverse_kinematics_shoulder_roll_and_pitch():    
    np.random.seed(42)

    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    # Shoulder roll and pitch:
    theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=position_right_shoulder_standard,
        rotation_right_shoulder_standard=rotation_right_shoulder_standard,
        position_right_elbow_inertial=rotation_right_shoulder_standard.apply(np.array([-0.5, -0.5, 1/np.sqrt(2)])) + position_right_shoulder_standard,
    )
    assert np.allclose(theta_r, -np.pi / 4)
    assert np.allclose(theta_p, np.pi / 4)


def test_right_shoulder_forward_then_inverse_kinematics_random_pitch():
    np.random.seed(43)
    
    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    theta_right_shoulder_pitch = np.random.random() * 2 * np.pi - np.pi
    theta_right_shoulder_roll = 0

    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=theta_right_shoulder_pitch,
            theta_right_shoulder_roll=theta_right_shoulder_roll,
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            right_arm_length=1.0,
        )

    theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=position_right_shoulder_standard,
        rotation_right_shoulder_standard=rotation_right_shoulder_standard,
        position_right_elbow_inertial=position_right_elbow_standard,
    )

    assert np.isclose(theta_p, theta_right_shoulder_pitch)
    assert np.isclose(theta_r, theta_right_shoulder_roll)


def test_right_shoulder_forward_then_inverse_kinematics_random_roll():
    np.random.seed(42)
    
    position_right_shoulder_standard = np.random.random([3])
    rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    theta_right_shoulder_pitch = 0
    theta_right_shoulder_roll = np.random.random() * np.pi - np.pi / 2

    position_right_elbow_standard, rotation_right_elbow_standard = \
        ForwardKinematics.forward_kinematics_right_elbow(
            theta_right_shoulder_pitch=theta_right_shoulder_pitch,
            theta_right_shoulder_roll=theta_right_shoulder_roll,
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            right_arm_length=1.0,
        )

    theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
        position_right_shoulder_standard=position_right_shoulder_standard,
        rotation_right_shoulder_standard=rotation_right_shoulder_standard,
        position_right_elbow_inertial=position_right_elbow_standard,
    )

    assert np.isclose(theta_p, theta_right_shoulder_pitch)
    assert np.isclose(theta_r, theta_right_shoulder_roll)


def test_right_shoulder_forward_then_inverse_kinematics_random_pitch_and_roll():
    np.random.seed(12)
    
    for _ in range(10):
        # Arbitrary shoulder pose:
        position_right_shoulder_standard = np.random.random([3])
        rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

        theta_right_shoulder_pitch = np.random.random() * 2 * np.pi - np.pi
        theta_right_shoulder_roll = np.random.random() * np.pi - np.pi / 2
        fk_arm_length = 10 * np.random.random()

        position_right_elbow_standard, rotation_right_elbow_standard = \
            ForwardKinematics.forward_kinematics_right_elbow(
                theta_right_shoulder_pitch=theta_right_shoulder_pitch,
                theta_right_shoulder_roll=theta_right_shoulder_roll,
                position_right_shoulder_standard=position_right_shoulder_standard,
                rotation_right_shoulder_standard=rotation_right_shoulder_standard,
                right_arm_length=fk_arm_length,
            )

        theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
            position_right_shoulder_standard=position_right_shoulder_standard,
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            position_right_elbow_inertial=position_right_elbow_standard,
        )

        assert np.isclose(theta_p, theta_right_shoulder_pitch)
        assert np.isclose(theta_r, theta_right_shoulder_roll)


def test_right_shoulder_inverse_then_forward_kinematics_random():
    np.random.seed(12)
    for _ in range(10):
        # Arbitrary shoulder pose:
        position_right_shoulder_standard = np.random.random([3])
        rotation_right_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

        position_right_elbow_standard_initial = np.random.random([3])
        initial_arm_length = np.linalg.norm(position_right_elbow_standard_initial - position_right_shoulder_standard)
        fk_arm_length = 10 * np.random.random()

        theta_r, theta_p = InverseKinematics.inverse_kinematics_right_shoulder(
            position_right_shoulder_standard=np.array(position_right_shoulder_standard),
            rotation_right_shoulder_standard=rotation_right_shoulder_standard,
            position_right_elbow_inertial=position_right_elbow_standard_initial,
        )

        position_right_elbow_standard, rotation_right_elbow_standard = \
            ForwardKinematics.forward_kinematics_right_elbow(
                theta_right_shoulder_pitch=theta_p,
                theta_right_shoulder_roll=theta_r,
                position_right_shoulder_standard=position_right_shoulder_standard,
                rotation_right_shoulder_standard=rotation_right_shoulder_standard,
                right_arm_length=fk_arm_length,
            )

        # Un-normalize length:
        position_right_elbow_standard = position_right_shoulder_standard + (position_right_elbow_standard - position_right_shoulder_standard) * (initial_arm_length / fk_arm_length)

        # Give a relatively large tolerance as small errors in IK can add to relatively large differences here
        assert np.allclose(position_right_elbow_standard, position_right_elbow_standard_initial, atol=0.01)


@pytest.mark.xfail
def test_left_shoulder_fk():
    raise NotImplementedError()


@pytest.mark.xfail
def test_left_shoulder_ik():
    raise NotImplementedError()


def test_left_shoulder_forward_then_inverse_kinematics_random_pitch():
    np.random.seed(43)

    position_left_shoulder_standard = np.random.random([3])
    rotation_left_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    theta_left_shoulder_pitch = np.random.random() * 2 * np.pi - np.pi
    theta_left_shoulder_roll = 0

    position_left_elbow_standard, rotation_left_elbow_standard = \
        ForwardKinematics.forward_kinematics_left_elbow(
            theta_left_shoulder_pitch=theta_left_shoulder_pitch,
            theta_left_shoulder_roll=theta_left_shoulder_roll,
            position_left_shoulder_standard=position_left_shoulder_standard,
            rotation_left_shoulder_standard=rotation_left_shoulder_standard,
            left_arm_length=1.0,
        )

    theta_r, theta_p = InverseKinematics.inverse_kinematics_left_shoulder(
        position_left_shoulder_standard=position_left_shoulder_standard,
        rotation_left_shoulder_standard=rotation_left_shoulder_standard,
        position_left_elbow_inertial=position_left_elbow_standard,
    )

    assert np.isclose(theta_p, theta_left_shoulder_pitch)
    assert np.isclose(theta_r, theta_left_shoulder_roll)


def test_left_shoulder_forward_then_inverse_kinematics_random_roll():
    np.random.seed(42)

    position_left_shoulder_standard = np.random.random([3])
    rotation_left_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

    theta_left_shoulder_pitch = 0
    theta_left_shoulder_roll = np.random.random() * np.pi - np.pi / 2

    position_left_elbow_standard, rotation_left_elbow_standard = \
        ForwardKinematics.forward_kinematics_left_elbow(
            theta_left_shoulder_pitch=theta_left_shoulder_pitch,
            theta_left_shoulder_roll=theta_left_shoulder_roll,
            position_left_shoulder_standard=position_left_shoulder_standard,
            rotation_left_shoulder_standard=rotation_left_shoulder_standard,
            left_arm_length=1.0,
        )

    theta_r, theta_p = InverseKinematics.inverse_kinematics_left_shoulder(
        position_left_shoulder_standard=position_left_shoulder_standard,
        rotation_left_shoulder_standard=rotation_left_shoulder_standard,
        position_left_elbow_inertial=position_left_elbow_standard,
    )

    assert np.isclose(theta_p, theta_left_shoulder_pitch)
    assert np.isclose(theta_r, theta_left_shoulder_roll)


def test_left_shoulder_forward_then_inverse_kinematics_random_pitch_and_roll():
    np.random.seed(12)

    for _ in range(10):
        # Arbitrary shoulder pose:
        position_left_shoulder_standard = np.random.random([3])
        rotation_left_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

        theta_left_shoulder_pitch = np.random.random() * 2 * np.pi - np.pi
        theta_left_shoulder_roll = np.random.random() * np.pi - np.pi / 2
        fk_arm_length = 10 * np.random.random()

        position_left_elbow_standard, rotation_left_elbow_standard = \
            ForwardKinematics.forward_kinematics_left_elbow(
                theta_left_shoulder_pitch=theta_left_shoulder_pitch,
                theta_left_shoulder_roll=theta_left_shoulder_roll,
                position_left_shoulder_standard=position_left_shoulder_standard,
                rotation_left_shoulder_standard=rotation_left_shoulder_standard,
                left_arm_length=fk_arm_length,
            )

        theta_r, theta_p = InverseKinematics.inverse_kinematics_left_shoulder(
            position_left_shoulder_standard=position_left_shoulder_standard,
            rotation_left_shoulder_standard=rotation_left_shoulder_standard,
            position_left_elbow_inertial=position_left_elbow_standard,
        )

        assert np.isclose(theta_p, theta_left_shoulder_pitch)
        assert np.isclose(theta_r, theta_left_shoulder_roll)


def test_left_shoulder_inverse_then_forward_kinematics_random():
    np.random.seed(12)
    for _ in range(10):
        # Arbitrary shoulder pose:
        position_left_shoulder_standard = np.random.random([3])
        rotation_left_shoulder_standard = Rotation.from_rotvec(np.random.random([3]))

        position_left_elbow_standard_initial = np.random.random([3])
        initial_arm_length = np.linalg.norm(position_left_elbow_standard_initial - position_left_shoulder_standard)
        fk_arm_length = 10 * np.random.random()

        theta_r, theta_p = InverseKinematics.inverse_kinematics_left_shoulder(
            position_left_shoulder_standard=np.array(position_left_shoulder_standard),
            rotation_left_shoulder_standard=rotation_left_shoulder_standard,
            position_left_elbow_inertial=position_left_elbow_standard_initial,
        )

        position_left_elbow_standard, rotation_left_elbow_standard = \
            ForwardKinematics.forward_kinematics_left_elbow(
                theta_left_shoulder_pitch=theta_p,
                theta_left_shoulder_roll=theta_r,
                position_left_shoulder_standard=position_left_shoulder_standard,
                rotation_left_shoulder_standard=rotation_left_shoulder_standard,
                left_arm_length=fk_arm_length,
            )

        # Un-normalize length:
        position_left_shoulder_standard = position_left_shoulder_standard + (
                    position_left_elbow_standard - position_left_shoulder_standard) * (
                                                    initial_arm_length / fk_arm_length)

        # Give a relatively large tolerance as small errors in IK can add to relatively large differences here
        assert np.allclose(position_left_shoulder_standard, position_left_elbow_standard_initial, atol=0.01)


@pytest.mark.xfail
def test_right_elbow_fk():
    raise NotImplementedError()


@pytest.mark.xfail
def test_right_elbow_ik():
    raise NotImplementedError()


@pytest.mark.xfail
def test_right_elbow_ik_then_fk():
    raise NotImplementedError()
