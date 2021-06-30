import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa
from numpy import pi
from scipy.spatial.transform import Rotation

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

    return all_frames


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
    rotation_right_arm_standard = Rotation.from_rotvec(theta_r * rot_vec_r) * rotation_right_arm

    # We wish to take the right arm standard frame and rotate it about its y axis such that the z axis is parallel with
    # the vector between the left and right arm
    z_r_standard = rotation_right_arm_standard.apply([0, 0, 1])
    theta_r = -np.arccos(np.dot((position_right_arm - position_left_arm)/np.linalg.norm(position_right_arm - position_left_arm), z_r_standard))
    y_r_standard = rotation_right_arm_standard.apply([0, 1, 0])
    rotation_right_arm_standard = Rotation.from_rotvec(theta_r * y_r_standard) * rotation_right_arm_standard

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
    bvh_frames_plus_standard['RightArmStandard'] = (position_right_arm, rotation_right_arm_standard)
    bvh_frames_plus_standard['LeftArmStandard'] = (position_left_arm, rotation_left_arm_standard)

    return bvh_frames_plus_standard


def main():
    all_frames = get_bvh_frames('nao_gestures/demos/examples/rand_5.bvh')
    all_frames = [add_standard_frames(frames) for frames in all_frames]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    drawable_frames = [
        'Hips',
        # 'LeftShoulder',
        # 'RightShoulder',
        # 'LeftArm',
        # 'RightArm',
        'LeftArmStandard',
        'RightArmStandard',
        'LeftForeArm',
        'RightForeArm',
        'LeftHand',
        'RightHand',
        'LeftUpLeg',
        'RightUpLeg',
        'LeftLeg',
        'RightLeg',
        'LeftFoot',
        'RightFoot',
        'Spine',
        'Spine1',
        'Spine2',
        'Spine3',
        'Head',
    ]
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

        for frame_name, (position, rotation) in frames.items():
            if len(drawable_frames) == 0 or frame_name in drawable_frames:
                draw_frame(ax, frame_name, position, rotation, arrow_length=10)
        plt.pause(0.05)


if __name__ == '__main__':
    main()
