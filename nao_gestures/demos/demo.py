from nao_gestures import NaoBvhConverter, NaoGesturePlayer

if __name__ == "__main__":
    # Get a playable dataframe of gestures:
    mocap_data = NaoBvhConverter.read_mocap_data('nao_gestures/demos/sample.bvh')
    df_gestures = NaoBvhConverter.bvh_to_dataframe_of_nao_gestures(mocap_data)

    # Play gestures:
    player = NaoGesturePlayer(
        robot_ip="127.0.0.1",
        robot_port=9559,
        my_ip="127.0.0.1",
        my_port=1234,
    )
    player.play(df_gestures, speed=1.0)
