import argparse
import pickle

import rospy
from std_msgs.msg import String

from nao_gestures import NaoBvhConverter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh_file', type=str, default='nao_gestures/demos/examples/rand_21.bvh')
    parser.add_argument('--topic', type=str, default='gestures_bvh')
    args = parser.parse_args()

    mocap_data = NaoBvhConverter.read_mocap_data(args.bvh_file)

    pub = rospy.Publisher(args.topic, String)
    rospy.init_node('bvh_publisher', anonymous=True)
    msg = String()
    msg.data = pickle.dumps(mocap_data)
    pub.publish(msg)


if __name__ == '__main__':
    main()
