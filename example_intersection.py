import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

class Intersection:

    def __init__(self):
        # initialize the world
        w = World(dt, width = 90, height = 90, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.
        
        # action_space with shape (20, 2) containing all possible combinations of acceleration and steering degree
        acc_values = [0, 1, 2, 3]
        steering_values = [0, 0.2, 0.6, -0.2, -0.6]
        self.action_space = np.array([[a, s] for a in acc_values for s in steering_values])

        # The resulting self.state_space array has shape (6930, 3) and represents all possible states 
        # in a 3-dimensional state space where each state is represented by its x, y, and velocity values.
        x_coord = np.arange(0, 46)
        y_coord = np.arange(40, 91)
        velocities = np.arange(3, 20, 0.5)
        self.state_space = np.array([[x, y, v] for x in x_coord for y in y_coord for v in velocities])
        self.observation_space = self.state_space
    
    def get_reward(self, action):
        # return reward given current state and action taken
        car_pos = (self.car_x, self.car_y)
        ped_pos = (self.ped_x, self.ped_y)

        # Check for collision with a pedestrian
        if self.distance(car_pos, ped_pos) < 1:
            return -150

        # Check for collision with another car
        for car in self.cars:
            if car != self and self.distance(car_pos, (car.car_x, car.car_y)) < 1:
                return -100

        # Check for collision with a building
        if self.car_x < 0 or self.car_x > 46 or self.car_y < 40 or self.car_y > 91:
            return -75

        # Check for proximity to a pedestrian or car
        for car in self.cars:
            if car != self and self.distance(car_pos, (car.car_x, car.car_y)) < 5:
                return -20
        if self.distance(car_pos, ped_pos) < 5:
            return -20

        # Check for close proximity to a pedestrian or car
        for car in self.cars:
            if car != self and self.distance(car_pos, (car.car_x, car.car_y)) < 1:
                return -40
        if self.distance(car_pos, ped_pos) < 1:
            return -40

        # Default case, no collision or proximity
        return 0


    def step(self, action):
        # update the current state given the most recent action taken
        cur_x, cur_y, cur_v = self.cur_state

        # update the velocity
        acc, steer = action
        new_v = cur_v + 0.1 * acc - 0.5 * abs(cur_v) * steer ** 2
        new_v = max(min(new_v, 20), 3)  # clip velocity between 3 and 20 m/s

        # update the position
        new_x = cur_x + new_v * np.cos(np.arctan2(cur_y - 45, cur_x - 18))
        new_y = cur_y + new_v * np.sin(np.arctan2(cur_y - 45, cur_x - 18))

        # check for collisions
        reward = 0
        done = False
        # w.collision_exists(p1)
        if new_x < 0 or new_x > 45 or new_y < 40 or new_y > 90:
            reward = -75  # collision with a building
            done = True
        elif np.any(np.linalg.norm(self.cur_state[:2] - self.pedestrians, axis=1) < 1):
            reward = -150  # collision with a pedestrian
            done = True
        elif np.any(np.linalg.norm(self.cur_state[:2] - self.cars, axis=1) < 1):
            reward = -100  # collision with another car
            done = True
        elif np.any(np.linalg.norm(self.cur_state[:2] - self.pedestrians, axis=1) < 5) or \
                np.any(np.linalg.norm(self.cur_state[:2] - self.cars, axis=1) < 5):
            reward = -20  # close to a pedestrian or car
        elif np.any(np.linalg.norm(self.cur_state[:2] - self.cars, axis=1) < 1):
            reward = -40  # very close to a car

        self.cur_state = np.array([new_x, new_y, new_v])

        return self.cur_state, reward, done



    # def step(self, action):

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
    c1_pos = Point(21,10+20)
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

    brake_dist_reached = False

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
                print('k == 100')
            elif c1.center.y >= 60-20 and brake_dist_reached is False:
                brake_dist_reached = True
                print('Brake distance reached, 20 meters away from intersection')
                print('c1 speed = ', c1.speed)
                print('accelerating...')
                c1.set_control(0, 0)
            elif c1.center.y >= 90:
                print('90 reached, c1 speed = ', c1.speed)
            elif k == 200: # The first Car starts pushing the brake a little bit. The second Car starts turning right with some throttle.
                c1.set_control(0, -0.02) 
                print('k == 200')
            elif k == 325:
                print('k == 325')               
                c1.set_control(0, 0.8)
                c2.set_control(-0.45, 0.3)
            elif k == 367: # The second Car stops turning.
                print('k = 367')
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
                print('Final speed of c1 = ', c1.speed)
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

