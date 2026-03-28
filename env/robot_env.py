import pybullet as p
import pybullet_data
import math

class RobotEnv:

    def __init__(self):

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane = p.loadURDF("plane.urdf")

        # robot start
        self.robot = p.loadURDF("r2d2.urdf", [0,0,0.5])

        # warehouse pickup station
        self.goal = [6,0]

        # store previous distance for reward shaping
        self.prev_goal_dist = None

        self.prev_pos = [0,0,0.5]

        # warehouse shelves
        shelf_positions = [
            [2,1,0.5],[2,-1,0.5],
            [3,1,0.5],[3,-1,0.5],
            [4,1,0.5],[4,-1,0.5],
            [5,1,0.5],[5,-1,0.5]
        ]

        self.obstacles = []

        for pos in shelf_positions:
            shelf = p.loadURDF("cube_small.urdf", pos)
            self.obstacles.append(shelf)

    def get_state(self):

        robot_pos,_ = p.getBasePositionAndOrientation(self.robot)

        rx, ry, _ = robot_pos

        goal_dist = math.sqrt((rx-self.goal[0])**2 + (ry-self.goal[1])**2)

        obstacle_dist = 999

        for obs in self.obstacles:

            pos,_ = p.getBasePositionAndOrientation(obs)

            ox, oy, _ = pos

            dist = math.sqrt((rx-ox)**2 + (ry-oy)**2)

            obstacle_dist = min(obstacle_dist, dist)

        return [rx, ry, goal_dist, obstacle_dist]


    def step(self, action):

        position, orientation = p.getBasePositionAndOrientation(self.robot)

        x, y, z = position

        # robot movement
        if action == 0:
            x += 0.1
        elif action == 1:
            y += 0.1
        elif action == 2:
            y -= 0.1

        # draw robot path
        p.addUserDebugLine(self.prev_pos, [x,y,z], [1,0,0], 2)
        self.prev_pos = [x,y,z]

        p.resetBasePositionAndOrientation(self.robot, [x,y,z], orientation)

        for _ in range(10):
            p.stepSimulation()

        state = self.get_state()

        goal_distance = state[2]

        # -------------------------
        # Reward shaping
        # -------------------------

        reward = -0.1
        done = False

        if self.prev_goal_dist is None:
            self.prev_goal_dist = goal_distance

        # reward for moving closer to goal
        if goal_distance < self.prev_goal_dist:
            reward += 1

        self.prev_goal_dist = goal_distance

        # goal reached
        if goal_distance < 0.4:
            print("GOAL REACHED")
            reward = 100
            done = True

        # collision with shelf
        if state[3] < 0.3:
            reward = -100
            done = True

        return state, reward, done