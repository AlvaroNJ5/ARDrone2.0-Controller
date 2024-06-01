from pyardrone import ARDrone, at

drone = ARDrone()
drone.navdata_ready.wait()
drone.send(at.CONFIG('general:navdata_demo', True))
drone.emergency()
print(drone.navdata.demo.vbat_flying_percentage)