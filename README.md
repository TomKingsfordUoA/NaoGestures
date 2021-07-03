![](nao_gestures/demos/examples/rand_5_sidebyside.mp4)

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

The naoqi Python SQK must be installed (see http://doc.aldebaran.com/2-1/dev/python/install_guide.html#python-install-guide).

The naoqi Python SDK should be extracted and its extracted directory added to the `PYTHONPATH` environment variable.

It is assumed the Nao robot is available either in simulation (i.e. as a virtual robot) or as physical hardware. More 
information can be found at http://doc.aldebaran.com/2-5/index_dev_guide.html.

# Getting Started

A demo can be found in `nao_gestures/demos/demo.py`.

# Credits

_TODO_

# License

_TODO_