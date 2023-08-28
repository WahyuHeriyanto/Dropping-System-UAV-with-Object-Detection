from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, Battery, LocationGlobal, Attitude
from pymavlink import mavutil
from servo import *

import time
import math
import numpy as np 
import psutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--connect', default='/dev/ttyUSB0')
#parser.add_argument('--connect', default='tcp:127.0.0.1:5762')
args = parser.parse_args()


connection_string = args.connect

#-- Create the object
plane = Plane(connection_string)
    