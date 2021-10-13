# NaoGestures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

https://user-images.githubusercontent.com/86583157/124368704-b3e22400-dcb7-11eb-9818-8e545acdc0e2.mp4

# Introduction

This library aims to ease the realisation of non-verbal gestures on a Softbanks Robotics Nao robot.

It includes functionality to:
1. Play a series of joint rotations on a Nao robot
2. Convert a BVH (motion capture) file to a series of joint rotations on a Nao robot

Notably, there are 6DOF in the shoulder and elbow in the BVH format, and only 4DOF in the corresponding joints. 
Moreover, the Nao robot has fewer total DOF than a human (including in the shoulders and neck). As such, I have manually
tuned the mapping from BVH format to Nao joint rotations to maximise the naturalness of the gestures. This is still a
work in progress.

# Requirements

The naoqi Python SQK must be installed (see [naoqi Python SDK Install Guide](http://doc.aldebaran.com/2-1/dev/python/install_guide.html#python-install-guide)).

The naoqi Python SDK should be extracted and its extracted directory added to the `PYTHONPATH` environment variable.

It is assumed the Nao robot is available either in simulation (i.e. as a virtual robot) or as physical hardware. More 
information can be found at [Nao Dev Guide](http://doc.aldebaran.com/2-5/index_dev_guide.html).

# Getting Started

    # Install dependencies
    cd /path/to/nao_gestures
    ./lib/download.sh
    export PYTHONPATH=$(pwd):$(pwd)/lib/pynaoqi-python2.7-2.8.6.23-linux64-20191127_152327/lib/python2.7/site-packages

    # Create a virtual environment:
    sudo apt install python3-virtualenv
    virtualenv --python=python2.7 venv
    source venv/bin/activate

    python -m pip install .

    nao_gestures -h

# ROS Integration

Use `Dockerfile` and `docker-compose.yml` in support of ROS functionality. Alternatively, you can import this library 
into another project, if that project can provide the requisite ROS dependencies. Refer to the `Dockerfile` for a
definition of ROS dependencies.

`docker-compose.yml` is self-contained and spawns its own `roscore`. If this is undesirable, simply remove the 
`roscore` service.

For a demo of ROS functionality:
1. Install [docker-compose](https://docs.docker.com/compose/install/) (v2.0+)
2. Run a virtual robot: `"/opt/Softbank Robotics/Choregraphe Suite 2.8/bin/naoqi-bin" -p 9559 -b 0.0.0.0`
3. Connect to the virtual robot with [Choreographe](https://developer.softbankrobotics.com/nao6/naoqi-developer-guide/choregraphe-suite/choregraphe-suite-installation-guide)
for visualisation.
4. `docker-compose up --build`

# Known Limitations

1. The Nao robot has more restrictive joint rotations than a human ([Nao Joints](http://doc.aldebaran.com/2-1/family/robots/joints_robot.html)). Where the inverse kinematics requests a joint rotation that exceeds the limits of the Nao robot, the rotation is merely clipped to the limit. This can result in, for example, grossly incorrect hand positions, in order to most faithfully achieve each individual joint rotation. In general, the tradeoff between the accuracy of hand position and joint angles is not clear and should be handled by the upstream module producing gestures.
2. Only elbow and shoulder joints are considered at this time.
