import argparse

from nao_gestures import NaoBvhConverter
from nao_gestures.nao_gesture_player import NaoqiNaoGesturePlayer, RosNaoGesturePlayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh_file', type=str, default='nao_gestures/demos/examples/rand_21.bvh')
    parser.add_argument('--dest_ip', type=str, default='127.0.0.1', help='Nao robot IP or ROS master IP')
    parser.add_argument('--dest_port', type=int, default=9559, help='Nao robot port or ROS master port')
    parser.add_argument('--my_ip', type=str, default='127.0.0.1', help='This machine\'s IP; can be a hostname for ROS')
    parser.add_argument('--my_port', type=int, default=1234, help='Naoqi port to use on this machine; unused for ROS')
    parser.add_argument('--use_ros', action='store_true')
    args = parser.parse_args()

    # Play gestures:
    if args.use_ros:
        player = RosNaoGesturePlayer(
            ros_master_uri='http://' + args.dest_ip + ':' + str(args.dest_port),
            my_ip_or_hostname=args.my_ip,
        )
        player.run()
    else:
        # Get a playable dataframe of gestures:
        mocap_data = NaoBvhConverter.read_mocap_data(args.bvh_file)
        df_gestures = NaoBvhConverter.bvh_to_dataframe_of_nao_gestures(mocap_data)

        player = NaoqiNaoGesturePlayer(
            robot_ip=args.dest_ip,
            robot_port=args.dest_port,
            my_ip=args.my_ip,
            my_port=args.my_port,
        )
        player.play(df_gestures, speed=1.0)


if __name__ == "__main__":
    main()
