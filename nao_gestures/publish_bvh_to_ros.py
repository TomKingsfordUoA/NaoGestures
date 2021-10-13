import argparse
import pickle
import json

import rospy
from std_msgs.msg import String

from nao_gestures import NaoBvhConverter
from nao_gestures.mocap_data_helpers import mocap_data_to_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh_file', type=str, default='nao_gestures/demos/examples/rand_21.bvh')
    parser.add_argument('--topic', type=str, default='gestures_bvh')
    args = parser.parse_args()

    mocap_data = NaoBvhConverter.read_mocap_data(args.bvh_file)

    pub = rospy.Publisher(args.topic, String)
    rospy.init_node('bvh_publisher', anonymous=True)
    msg = String()
    msg.data = json.dumps(mocap_data_to_json(mocap_data))
    pub.publish(msg)


if __name__ == '__main__':
    main()
