import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

class Intersection:

    def __init__(self):
        self.action_space = []
        self.state_space = []
        self.observation_space = []

    human_controller = False

    dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
    w = World(dt, width = 90, height = 90, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
    # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
    # For both of these objects, we give the center point and the size.
    w.add(Painting(Point(71.5, 106.5 - 20), Point(97, 27), 'purple')) # We build a sidewalk.
    w.add(RectangleBuilding(Point(72.5, 107.5 - 20), Point(95, 25))) # The RectangleBuilding is then on top of the sidewalk, with some margin.
    # w.add(RectangleBuilding(Point(72.5, 107.5), Point(5, 5))) # The RectangleBuilding is then on top of the sidewalk, with some margin.


    # Let's repeat this for 4 different RectangleBuildings.
    w.add(Painting(Point(7.5, 106.5 - 20), Point(17, 27), 'green'))
    w.add(RectangleBuilding(Point(6.5, 107.5 - 20), Point(15, 25)))

    w.add(Painting(Point(7.5, 41 - 20), Point(17, 82), 'orange'))
    w.add(RectangleBuilding(Point(6.5, 40 - 20), Point(15, 80)))

    w.add(Painting(Point(71.5, 41 - 20), Point(97, 82), 'yellow'))
    w.add(RectangleBuilding(Point(72.5, 40 -20), Point(95, 80)))

    # Let's also add some zebra crossings, because why not.
    w.add(Painting(Point(18, 81-20), Point(0.5, 2), 'white'))
    w.add(Painting(Point(19, 81-20), Point(0.5, 2), 'white'))
    w.add(Painting(Point(20, 81-20), Point(0.5, 2), 'white'))
    w.add(Painting(Point(21, 81-20), Point(0.5, 2), 'white'))
    w.add(Painting(Point(22, 81-20), Point(0.5, 2), 'white'))



    # INITIALIZE RANDOM VARIABLES
    # C1
    # c1_pos = Point(21,50)
    # c1_pos = Point(120,120)
    c1_pos = Point(21,10+30)
    c1_vel = Point(3.0, 0)

    # C2
    c2_pos = Point(77,90 - 20)
    c2_vel = Point(4.0,0)

    # C3
    c3_pos = Point(18,98 - 20)
    c3_vel = Point(0, 0)



    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    c1 = Car(c1_pos, np.pi/2) # Point(20,20)
    c1.velocity = c1_vel
    w.add(c1)

    c2 = Car(c2_pos, np.pi, 'blue')
    c2.velocity = c2_vel # We can also specify an initial velocity just like this. Point(3.0,0)
    w.add(c2)

    c3 = Car(c3_pos, np.pi*3/2, 'yellow')
    c3.velocity = c3_vel # We can also specify an initial velocity just like this. Point(3.0,0)
    w.add(c3)

    # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
    p1 = Pedestrian(Point(28,81-20), np.pi)
    # p1.velocity = Point(2.0, 0)
    p1.max_speed = 10.0 # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
    w.add(p1)

    w.render() # This visualizes the world we just constructed.


    if not human_controller:
        # Let's implement some simple scenario with all agents
        p1.set_control(0, 0.32) # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
        # c1.set_control(0, 0.35)
        c1.set_control(0,0)
        c2.set_control(0, 0.05)
        for k in range(400):
            # All movable objects will keep their control the same as long as we don't change it.
            if k == 100: # Let's say the first Car will release throttle (and start slowing down due to friction)
                c1.set_control(0, 0)
            elif k == 200: # The first Car starts pushing the brake a little bit. The second Car starts turning right with some throttle.
                c1.set_control(0, -0.02) 
            elif k == 325:
                c1.set_control(0, 0.8)
                c2.set_control(-0.45, 0.3)
            elif k == 367: # The second Car stops turning.
                c2.set_control(0, 0.1)
            w.tick() # This ticks the world for one time step (dt second)
            w.render()
            time.sleep(dt/4) # Let's watch it 4x

            if w.collision_exists(p1): # We can check if the Pedestrian is currently involved in a collision. We could also check c1 or c2.
                print('Pedestrian has died!')
                # TODO: slow down speed of car according to momentum loss from hitting the pedestrian
            elif w.collision_exists(): # Or we can check if there is any collision at all.
                print('Collision exists somewhere...')
                time.sleep(4)
                break

            # terminal condition
            if c1.center.y > 119:
                print('Road cleared, collisioned avoided!')
                time.sleep(4)
                break
        w.close()

    else: # Let's use the steering wheel (Logitech G29) for the human control of car c1
        p1.set_control(0, 0.22) # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
        c2.set_control(0, 0.35)
        
        from interactive_controllers import SteeringWheelController
        controller = SteeringWheelController(w)
        for k in range(400):
            c1.set_control(controller.steering, controller.throttle)
            w.tick() # This ticks the world for one time step (dt second)
            w.render()
            time.sleep(dt/4) # Let's watch it 4x
            if w.collision_exists():
                import sys
                sys.exit(0)

