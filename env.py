import datetime
import gymnasium as gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import time
import yaml

import EnvCreator
import path_planning

class SearchEnv(gym.Env):
    def __init__(self, training=True, cfg=None):
        super().__init__()
        self.date = datetime.datetime.now().strftime("%m%d_%H%M")
        self.training = training
        if isinstance(cfg,str):
            assert os.path.exists(cfg), "specified configuration file does not exist"
            with open(cfg, 'r') as stream:
                self.cfg = yaml.safe_load(stream)
        else:
            raise AssertionError("no configuration file specified")

        self.rng = np.random.default_rng(seed=self.cfg["SEED"])

        if self.training:
            self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else:
            self.client = bullet_client.BulletClient(connection_mode=p.GUI)
            self.client.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

        self.action_space = gym.spaces.MultiDiscrete(nvec=[self.cfg["N_ACTIONS"] for _ in range(self.cfg["N_ROBOTS"])], dtype=np.int32, seed=self.rng)
        self.observation_space = gym.spaces.Dict({
            "data"    : gym.spaces.Box(low=-1, high=1, shape=(self.cfg["N_CHANNELS"], self.cfg["HEIGHT"], self.cfg["WIDTH"]), dtype=np.float32, seed=self.rng),
            "context" : gym.spaces.Box(low=-1, high=1, shape=(self.cfg["N_ROBOTS"] * self.cfg["CONTEXT"],), dtype=np.float32, seed=self.rng)
            }, seed=self.rng)

    def reset(self, seed=None, options=None):
        ## initiate simulation
        self.client.resetSimulation()
        self.client.setTimeStep(1./self.cfg["PHYSICS_HZ"])
        self.client.setGravity(0, 0, -10)

        ## setup ground
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = self.client.loadURDF("plane.urdf")
        self.client.changeDynamics(self.ground, -1, lateralFriction=0.0,
                spinningFriction=0.0, rollingFriction=0.0)

        ## setup walls
        random_idx = self.rng.integers(0, len(self.cfg["MAP_LIST"]))
        file = self.cfg["MAP_LIST"][random_idx]
        env_c = EnvCreator.envCreator(file=file,resolution=self.cfg["RESOLUTION"],height=2,density=1,flip=False)
        self.map = env_c.image2occupancy()
        map_urdf = env_c.get_urdf_fast(output_dir="./maps")
        self.walls = self.client.loadURDF(map_urdf, basePosition=[0, -9.1, 0], useFixedBase=True)#, flags=p.URDF_MERGE_FIXED_LINKS)
        self.client.changeDynamics(self.walls, -1, lateralFriction=0.0,
                spinningFriction=0.0, rollingFriction=0.0)

        ## setup map and low dim map
        self.entropy = np.zeros_like(self.map, dtype=np.float32)
        k = self.cfg["MAP_REDUCTION_FACTOR"]
        mini_rows = self.cfg["HEIGHT"] // k
        mini_cols = self.cfg["WIDTH"] // k
        self.mini_map = self.map.reshape(mini_rows, k, mini_cols, k).max(axis=(1, 3))
        self.mini_map_grid = path_planning.Grid(self.mini_map)

        ##setup robots
        self.robots = []
        self.competancies = {}
        for n in range(self.cfg["N_ROBOTS"]):
            pose,ori = self._get_random_pose()
            robot = self.client.loadURDF(self.cfg["ROBOT_URDF"],
                basePosition=[pose[0], pose[1], 0.25], 
                baseOrientation=p.getQuaternionFromEuler([0, 0, ori]))
            self.client.changeDynamics(robot, -1, lateralFriction=0.0,
                spinningFriction=0.0, rollingFriction=0.0)
            self.robots.append(robot)
            vel = self.rng.random()
            fov = self.rng.random()
            lidar_range = self.rng.random()
            self.competancies[n] = {
                "Vel"   : vel,
                "FOV"   : fov,
                "Range" : lidar_range
            }

        ##setup targets
        self.targets = []
        self.target_nodes = []
        for n in range(self.cfg["N_TARGETS"]):
            pose,_ = self._get_random_pose()
            target = self.client.loadURDF(self.cfg["TARGET_URDF"],
                useFixedBase=True,
                basePosition=[pose[0], pose[1], 0.25], 
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
            self.targets.append(target)
            target_node = path_planning.Node()
            target_node.set_pose(self._rc2minirc(*self._xy2rc(*pose[:2])))
            self.target_nodes.append(target_node)

        ## initialize objects
        for _ in range(10):
            self.client.stepSimulation()

        ## initialize tracking vars
        self.detection = False
        self.collision = False
        self.done = False
        self.t = 0
        self.target_idx = 0
        self.repeat = max(1,self.cfg["PHYSICS_HZ"] // self.cfg["UPDATE_HZ"])

        self.obs = self._get_obs()
        self._setup_plot()

        return self.obs, {}

    def step(self, action):
        self.t += self.repeat
        self.path_lens = []
        vels = []
        for i in range(len(self.robots)):
            context_length = self.obs["context"].shape[0] // self.cfg["N_ROBOTS"]
            pose = np.array([self.obs["context"][context_length*i + 0], self.obs["context"][context_length*i + 1]])
            pose_rc_mini = self._rc2minirc(*self._xy2rc(*pose))
            th = np.pi / 4 * action[i]
            waypt = self.cfg["WAYPOINT_RADIUS"] * np.array([np.cos(th), np.sin(th)]) + pose
            pose_node = path_planning.Node()
            pose_node.set_pose(pose_rc_mini)
            waypt_node = path_planning.Node()
            waypt_node.set_pose(self._rc2minirc(*self._xy2rc(*waypt)))
            path = path_planning.A_Star(self.mini_map_grid, pose_node, waypt_node)
            if path is None:
                self.path_lens.append(self.cfg["WAYPOINT_ERROR_COST"])
                vels.append(np.array([0, 0, 0]))
                if not self.training: print("Bad waypoint.")
            else:
                self.path_lens.append(0.5 * len(path)) #mini_map blocks are 50cm
                next_waypt = path[1] #skip starting pose
                direction = np.array([pose_rc_mini[1] - next_waypt[1], -pose_rc_mini[0] + next_waypt[0], 0])
                vel = -direction / np.linalg.norm(direction)
                vels.append(vel)

        for i in range(self.repeat):
            for j in range(len(self.robots)):
                vel_scale = (self.cfg["MAX_VEL"] - self.cfg["MIN_VEL"]) * self.competancies[j]["Vel"] + self.cfg["MIN_VEL"]
                vel_err = vel_scale * vels[j] - np.array(self.client.getBaseVelocity(self.robots[j])[0])
                force = self.cfg["K_V"] * np.array([vel_err[0], vel_err[1], 0])
                #self.client.applyExternalForce(self.robots[j], -1, force, [0,0,0], p.LINK_FRAME)
                self.client.applyExternalForce(self.robots[j], -1, force, self.client.getBasePositionAndOrientation(self.robots[j])[0], p.WORLD_FRAME)
                
                th = p.getEulerFromQuaternion(self.client.getBasePositionAndOrientation(self.robots[j])[1])[2]
                th_d = np.arctan2(vels[j][1],vels[j][0])
                #th_err = self._wrap_angle(th_d) - self._wrap_angle(th)
                th_err = self._wrap_angle(th_d-th)
                th_dot_err = 0 - self.client.getBaseVelocity(self.robots[j])[1][2]
                torque = self.cfg["K_P_TH"] * th_err + self.cfg["K_V_TH"] * th_dot_err
                self.client.applyExternalTorque(self.robots[j], -1, [0, 0, torque], p.WORLD_FRAME)

                ## collision detection
                ctx = self.client.getContactPoints(self.robots[j],self.walls)
                if len(ctx) > 0:
                    self.collision = True
            self.client.stepSimulation()
            if not self.training:
                time.sleep(1./self.cfg["PHYSICS_HZ"])

        self.obs = self._get_obs()
        self.rew = self._get_rew()
        self.done = self._get_dones()
        self.infos = {"terminal_observation" : self.obs} if self.done else {}
        self._plot()
        
        return self.obs, self.rew, self.done, self.t >= self.cfg["MAX_STEPS"], self.infos

    def render(self):
        pass

    def close(self):
        self.client.disconnect()

    def _get_obs(self):
        ## obs = {data,context}
        ## data = entropy
        ## context = [*robot_poses,*coverage_competancies]
        ## coverage_competancy = [max_v,FOV,range]
        self.last_entropy = self.entropy.copy()
        self.entropy = self._decay_entropy(self.entropy)
        self.entropy = self._get_scans(self.entropy)
        context = []
        for i in range(len(self.robots)):
            pose,oriq = self.client.getBasePositionAndOrientation(self.robots[i])
            ori = p.getEulerFromQuaternion(oriq)[2]
            context.append(pose[0] / (0.5 * self.cfg["WIDTH"] * self.cfg["RESOLUTION"]))
            context.append(pose[1] / (0.5 * self.cfg["HEIGHT"] * self.cfg["RESOLUTION"]))
            context.append(ori / np.pi)
            for c in self.competancies[i].values():
                context.append(c)
        obs = {
        "data"    : self.entropy.reshape((1, self.cfg["HEIGHT"], self.cfg["WIDTH"])),
        "context" : np.array(context, dtype=np.float32)
        }
        return obs

    def _get_rew(self):
        info_gain_rew = np.sum(np.abs(self.entropy)) - np.sum(np.abs(self.last_entropy))
        feasability_rew = 0
        separation_rew = 0
        for i in range(len(self.robots)):
            feasability_rew -= self.path_lens[i] - self.cfg["WAYPOINT_RADIUS"]
            for j in range(len(self.robots)):
                if i != j:
                    pose1, _ = self.client.getBasePositionAndOrientation(self.robots[i])
                    pose2, _ = self.client.getBasePositionAndOrientation(self.robots[j])
                    separation_rew += np.linalg.norm(np.array(pose2[:2]) - np.array(pose1[:2])) / (2**0.5 * self.cfg["WIDTH"] * self.cfg["RESOLUTION"])
        separation_rew /= 2 #poses are double counted
        collision_rew = -self.cfg["COLLISION_COST"] * self.collision
        detection_rew = self.cfg["DETECTION_REWARD"] * self.detection
        #print(info_gain_rew)
        #print(feasability_rew)
        #print(separation_rew)
        #print(collision_rew)
        #print(detection_rew)
        #print()

        return info_gain_rew + feasability_rew + separation_rew + collision_rew + detection_rew

    def _get_dones(self):
        return self.collision or self.detection# or self.t >= self.cfg["MAX_STEPS"]

    def _get_scans(self, entropy):
        origins = []
        endpts = []
        robot_ids = []
        for i in range(len(self.robots)):
            range_scale = (self.cfg["MAX_RANGE"] - self.cfg["MIN_RANGE"]) * self.competancies[i]["Range"] + self.cfg["MIN_RANGE"]
            fov_scale = (self.cfg["MAX_FOV"] - self.cfg["MIN_FOV"]) * self.competancies[i]["FOV"] + self.cfg["MIN_FOV"]
            fov_rad = 2 * np.pi * fov_scale / 360
            n_scans = int(fov_rad / np.arctan(self.cfg["RESOLUTION"] / range_scale)) 
            scan_angle = fov_rad / n_scans
            scan_height = 1.0
            pose,oriq = self.client.getBasePositionAndOrientation(self.robots[i])
            ori = p.getEulerFromQuaternion(oriq)[2]
            origin = [[pose[0], pose[1], scan_height] for j in range(n_scans)]
            endpt = [
            [
                range_scale * np.cos(j * scan_angle + ori - fov_rad / 2) + pose[0],
                range_scale * np.sin(j * scan_angle + ori - fov_rad / 2) + pose[1],
                scan_height
            ]
                for j in range(n_scans)]
            origins += origin
            endpts += endpt
            robot_ids += [self.robots[i]] * n_scans
        
        scan_data = self.client.rayTestBatch(origins,endpts,numThreads=1)
        for i in range(len(robot_ids)):
            hit_idx = []
            hit = False
            if scan_data[i][0] == -1:
                hr, hc = self._xy2rc(*endpts[i][:2])
            else:
                hit = True
                hr, hc = self._xy2rc(*scan_data[i][3][:2])
                hit_idx.append([hr, hc])
                if scan_data[i][0] in self.targets:
                    self.detection = True
                    #print("Target found.")
            pose, _ = self.client.getBasePositionAndOrientation(robot_ids[i])
            sr, sc = self._xy2rc(*pose[:2])
            free_idx = self._bresenham((sr, sc), (hr, hc))
            if hit:
                free_idx = free_idx[:-1]
            for f in free_idx:
                entropy[f[0], f[1]] = -1
            for h in hit_idx:
                entropy[h[0], h[1]] = 1
        return entropy

    def _decay_entropy(self, entropy):
        return entropy * self.cfg["DECAY_RATE"]

    def _setup_plot(self):
        if not self.training:
            try:
                plt.close("all")
            except:
                pass
            self.fig,self.ax = plt.subplots()
            self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
            self.ax.set_axis_off()
            self.frames = []
            self.fps = int(self.cfg["PHYSICS_HZ"])
            self.writer = animation.FFMpegWriter(fps=self.fps) 

    def _plot(self):
        if not self.training:
            self.frames.append([self.ax.imshow(self.entropy, animated=True, vmin=-1, vmax=1)])
            if (self.done or self.t >= self.cfg["MAX_STEPS"]):
                ani = animation.ArtistAnimation(self.fig, self.frames, interval=int(1000 / self.fps), blit=True, repeat=False)
                plt.show()
                #ani.save(PATH_DIR+self.log_dir+self.log_name,writer=self.writer)

    def _get_random_pose(self,mark=False,idx=1):
        k = self.cfg["MAP_REDUCTION_FACTOR"]
        free_idx = np.argwhere(self.mini_map == 0)
        mini_r, mini_c = self.rng.choice(free_idx)
        r = k * mini_r + (k // 2)
        c = k * mini_c + (k // 2)
        if mark:
            self._set_mini_map(r,c,idx)
        th = 2 * np.pi * self.rng.random(dtype=np.float32)
        return self._rc2xy(r, c), th

    def _set_mini_map(self,r,c,value):
        k = self.cfg["MAP_REDUCTION_FACTOR"]
        rr = r // k
        cc = c // k
        self.mini_map[rr,cc] = value

    def _rc2xy(self, r, c):
        r0 = self.cfg["HEIGHT"] / 2
        c0 = self.cfg["WIDTH"] / 2
        rr = r0 - r #origin at center
        cc = c - c0 #origin at center
        return self.cfg["RESOLUTION"] * np.array([cc, rr])

    def _xy2rc(self, x, y):
        r0 = self.cfg["HEIGHT"] / 2
        c0 = self.cfg["WIDTH"] / 2
        rc = np.array([x, -y]) / self.cfg["RESOLUTION"] + np.array([c0, r0])
        return int(rc[1]), int(rc[0])

    def _rc2minirc(self, r, c):
        k = self.cfg["MAP_REDUCTION_FACTOR"]
        return r // k, c // k

    def _wrap_angle(self, th):
        th = (th + 2 * np.pi) % (2 * np.pi)
        if th > np.pi:
            th -= 2 * np.pi
        return th

    def _bresenham(self, start, end):
        """
        Adapted from PythonRobotics:
        https://github.com/AtsushiSakai/PythonRobotics/blob/master/Mapping/lidar_to_grid_map/lidar_to_grid_map.py

        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a list from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        [[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]]
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        return points#[1:-1] #do not include endpoints

if __name__ == "__main__":
    #from stable_baselines3.common.env_checker import check_env
    
    env = SearchEnv(training=False, cfg="./base_config.yaml")
    #check_env(env)
    env.reset()

    import time
    for t in range(10_000):
        obs, rew, done, infos = env.step([1/2,-1/2,0])
        if done: break
        time.sleep(1./100.)