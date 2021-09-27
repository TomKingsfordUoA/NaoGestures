#!/bin/bash

cd "$(dirname "$0")/.."
set -e

NAOQI_FILE=pynaoqi-python2.7-2.8.6.23-linux64-20191127_152327.tar.gz
wget -O ./lib/${NAOQI_FILE} https://community-static.aldebaran.com/resources/2.8.6/${NAOQI_FILE}
tar --directory=$(pwd)/lib -xzf ./lib/${NAOQI_FILE}
