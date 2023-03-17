import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

human_controller = False


# State space = Rows by columns grid, initial position of cars, initial speed of cars, number of passengers, number of lanes

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(91.75,60), np.pi/2)
c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)


# State uncertainty of the car: what its sensor percieves about the environment

# 1) Front sensor only, 200 meters range, 120° horizontally

# 2) Side sensors (shorter range)

