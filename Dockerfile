FROM ros:melodic-ros-core

# Core system:
RUN apt update
RUN apt install -y python3 python3-pip python python-pip
RUN apt install -y ros-melodic-ros-base

# Specific requirements:
RUN apt install -y ros-melodic-rospy ros-melodic-naoqi-bridge-msgs
RUN python3 -m pip install rospkg
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN apt install -y ros-melodic-naoqi-driver

WORKDIR /nao_gestures
COPY ./nao_gestures /nao_gestures/nao_gestures
COPY ./pymo /nao_gestures/pymo
COPY ./docker-entrypoint.sh /nao_gestures/docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["python", "nao_gestures/cli.py", "-h"]
