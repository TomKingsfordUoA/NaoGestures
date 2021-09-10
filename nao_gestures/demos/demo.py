import argparse

from nao_gestures import NaoBvhConverter, NaoGesturePlayer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh_file', type=str, default='nao_gestures/demos/examples/rand_21.bvh')
    parser.add_argument('--robot_ip', type=str, default='127.0.0.1')
    parser.add_argument('--robot_port', type=int, default=9559)
    parser.add_argument('--my_ip', type=str, default='127.0.0.1')
    parser.add_argument('--my_port', type=int, default=1234)
    args = parser.parse_args()

    # Get a playable dataframe of gestures:
    mocap_data = NaoBvhConverter.read_mocap_data(args.bvh_file)
    df_gestures = NaoBvhConverter.bvh_to_dataframe_of_nao_gestures(mocap_data)

    # Play gestures:
    player = NaoGesturePlayer(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        my_ip=args.my_ip,
        my_port=args.my_port,
    )
    player.play(df_gestures, speed=1.0)
