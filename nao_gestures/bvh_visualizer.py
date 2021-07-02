import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa

from nao_gestures import NaoGesturePlayer
from nao_gestures.bvh_converter import NaoBvhConverter
from nao_gestures.kinematics import InverseKinematics, ForwardKinematics


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


def main():
    mocap_data = NaoBvhConverter.read_mocap_data('nao_gestures/demos/examples/rand_5.bvh')
    all_frames, index = NaoBvhConverter.get_bvh_frames(mocap_data)
    all_frames = [NaoBvhConverter.add_standard_frames(frames) for frames in all_frames]
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
