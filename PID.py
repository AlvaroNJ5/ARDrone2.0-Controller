import numpy as np

dist_x_ant = 0
dist_y_ant = 0
ratio_ant = 1
kp = 1
kd = 0.9
ki = 0
pid = [kp, kd, ki]

def PID_control(area_ref, area, cx, cy):
    global dist_x_ant, dist_y_ant, ratio_ant
    adat = 0
    arab = 0
    id = 0
    #Control de adelante (ad) - atrás (at)
    ratio = area / area_ref
    if abs(ratio - ratio_ant) > 0.1:
        adat = pid[0] * (ratio - 1) + pid[1] * (ratio - ratio_ant)
        adat = np.clip(adat, -0.3, 0.3)
    ratio_ant = ratio

    #Control de arriba (ar) - abajo (ab)
    dist_y = 224 - cy
    if abs(dist_y) >= 30:
        arab = pid[0] * dist_y + pid[1] * (dist_y - dist_y_ant)
        arab = np.clip(arab, -0.15, 0.15)
    dist_y_ant = dist_y
    
    #Control de izquierda (i) - derecha (d)
    dist_x = 224 - cx
    if dist_x != 0:
        id = pid[0] * dist_x + pid[1] * (dist_x - dist_x_ant)
        id = np.clip(id, -0.1, 0.1)
    dist_x_ant = dist_x

    return adat, arab, id
