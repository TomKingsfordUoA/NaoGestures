#!/bin/bash
set -e

source /opt/ros/melodic/setup.sh
export PYTHONPATH=$(pwd):$PYTHONPATH
exec "$@"
