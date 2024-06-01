import cv2
from pyardrone import ARDrone, at
import time
import numpy as np

import logging



def init_drone():

    #logging.basicConfig(level=logging.DEBUG)
    drone = ARDrone()
    drone.navdata_ready.wait()
    drone.video_ready.wait()
    drone.send(at.CONFIG('general:navdata_demo', True))
    time.sleep(1)
    #drone.move()   #todas las velocidades a 0
    return drone


def get_battery(drone):
    return drone.navdata.demo.vbat_flying_percentage

def data_print(drone):
    battery = drone.navdata.demo.vbat_flying_percentage
    altitude = drone.navdata.demo.altitude
    theta = drone.navdata.demo.theta
    phi = drone.navdata.demo.phi
    psi = drone.navdata.demo.psi
    vx = drone.navdata.demo.vx
    vy = drone.navdata.demo.vy
    vz = drone.navdata.demo.vz
    print(f'DRONE INFO:\n\tBATTERY: {battery}\n\tALTITUDE: {altitude}\n\tTHETA: {theta}\n\tPHI: {phi}\n\tPSI: {psi}\n\tVX: {vx}\n\tVY: {vy}\n\tVZ: {vz}\n\t')
def drone_get_frame(drone, w=360, h=240):

    img = cv2.resize(drone.frame, (w,h))
    return img

def takeoff_control(drone):
    init_battery = get_battery(drone)
    if init_battery > 25:
        while not drone.state.fly_mask:
            drone.takeoff()
        drone.hover()
        time.sleep(1)

def findFace(img):
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 4)
    aux = 0
    for (x,y,w,h) in faces:
        aux = 1
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0),2)
        cx = x + w//2
        cy = y + h//2
        area = w*h
    if aux == 1:
        return img, [[cx,cy], area]
    else:
        return img, [[0,0],0]


def trackFace(drone, cx, w, pid, p_error):
    'El objetivo es tener, con error 0, w//2 (320 en este caso)'
    a_error = cx - w//2
    speed = pid[0]*a_error + pid[1]*(a_error-p_error)
    speed = speed/100.0
    if speed > 0.3:
        speed = 0.3
    elif speed < -0.3:
        speed = -0.3

    if cx!= 0:
        drone.move(forward=0, backward=0, left=0, right=0, up=0, down=0, cw=speed, ccw=0)
    else:
        drone.hover()
        a_error = 0

    return a_error

def height_control(drone,cy, h, pid, p_error):
    a_error = cy - h//2
    speed = pid[0]*a_error + pid[1]*(a_error-p_error)
    speed = speed / 100.0
    if speed > 0.3:
        speed = 10.3
    elif speed < -0.3:
        speed = -0.3

    if cy != 0:
        print(speed)
        drone.move(forward=0, backward=0, left=0, right=0, up=-speed, down=0, cw=0, ccw=0)
    else:
        drone.hover()
        a_error = 0

    return a_error

def dist_control(drone,ar, area, pid, p_error):
    a_error = ar - area//2
    speed = pid[0] * a_error + pid[1] * (a_error - p_error)

    if speed > 0.3:
        speed = 0.3
    elif speed < -0.3:
        speed = -0.3

    if ar != 0:
        print(f'velocidad:{speed}, area: {ar}')
        drone.move(forward=-speed, backward=0, left=0, right=0, up=0, down=0, cw=0, ccw=0)
    else:
        drone.hover()
        a_error = 0
    return a_error