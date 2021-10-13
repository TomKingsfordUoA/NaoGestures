import json

import pandas as pd
import pymo.data


# TODO(TK): Consider replacing JSON with ROS message
def mocap_data_from_json(json_mocap_data):
    if isinstance(json_mocap_data, str):
        json_mocap_data = json.loads(json_mocap_data)
    mocap_data = pymo.data.MocapData()
    mocap_data.skeleton = json_mocap_data['skeleton']
    mocap_data.values = pd.read_json(json_mocap_data['values'], orient='split')
    mocap_data.channel_names = json_mocap_data['channel_names']
    mocap_data.framerate = json_mocap_data['framerate']
    mocap_data.root_name = json_mocap_data['root_name']

    mocap_data.values.reset_index(inplace=True)
    mocap_data.values.index *= mocap_data.framerate  # this is actually the frame period
    mocap_data.values.index = pd.to_timedelta(mocap_data.values.index, unit='S')

    return mocap_data


def mocap_data_to_json(mocap_data):
    return {
        'skeleton': mocap_data.skeleton,
        'values': mocap_data.values.to_json(orient='split'),
        'channel_names': mocap_data.channel_names,
        'framerate': mocap_data.framerate,
        'root_name': mocap_data.root_name,
    }