import numpy as np
import math

class DWA_Pursuit:
    def __init__(self):
        self.v_window = [0.0, 0.6]
        self.w_window = [-0.9, 0.9]
        self.acc_v = 5.0
        self.acc_w = 5.0
        self.resol_v = 0.1
        self.resol_w = 0.05
        self.dt = 0.4

        self.robot_r = 0.17+0.03
        self.inf_r = 0.2

        self.v_weight = 0.2
        self.g_weight = 1.0
        self.o_weight = 1.0

        self.last_v = 0.0
        self.last_w = 0.0

    def reset(self):
        self.last_v = self.last_w = 0.0

    def recovery_action(self):
        self.last_v = self.last_v-self.acc_v*self.dt if self.last_v-self.acc_v*self.dt >= self.v_window[0] else self.v_window[0]
        self.last_w = self.last_w+self.acc_w*self.dt if self.last_w+self.acc_w*self.dt <= self.w_window[1] else self.w_window[1]
        return [self.last_v, self.last_w]

    def scan_to_obs(self, scan, min_angle=-math.pi/2, max_angle=math.pi/2, max_range=4, filter_scale = 4):
        obs_list = []
        range_total = scan.shape[0] // filter_scale
        angle_step = (max_angle - min_angle) / range_total
        cur_angle = min_angle
        for i in range(range_total):
            j = i * filter_scale
            if scan[j] < max_range:
                x = scan[j] * math.cos(cur_angle)
                y = scan[j] * math.sin(cur_angle)
                obs_list.append([x,y])
            cur_angle += angle_step
        return obs_list

    def calc_dynamic_window(self):
        vs = self.v_window
        vs.extend(self.w_window)
        vd = [self.last_v - self.acc_v * self.dt,
              self.last_v + self.acc_v * self.dt,
              self.last_w - self.acc_w * self.dt,
              self.last_w + self.acc_w * self.dt]
        vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]
        return vr

    def calc_reach(self, vt, wt):
        reach = [0.0, 0.0, 0.0]
        if wt != 0.0:
            reach[0] += vt/wt * math.sin(wt*self.dt)
            reach[1] += vt/wt * (1 - math.cos(wt*self.dt))
            reach[2] += (wt * self.dt) % (2 * math.pi)
        else:
            reach[0] += vt * self.dt
        return reach

    def cost_v_w(self, vt, wt):
        return abs(vt-self.last_v) + 0.2*abs(wt-self.last_w)

    def cost_goal(self, point, traj):
        dist = math.inf
        yaw = point[2]
        for i in range(len(traj)-1):
            _v0 = (traj[i+1][0]-traj[i][0], traj[i+1][1]-traj[i][1])
            _v1 = (point[0]-traj[i][0], point[1]-traj[i][1])
            _v2 = (point[0]-traj[i+1][0], point[1]-traj[i+1][1])
            _check = (_v1[0]*_v0[0]+_v1[1]*_v0[1])*(_v2[0]*_v0[0]+_v2[1]*_v0[1])
            if _check > 0: continue
            dist = abs(_v1[0]*_v0[1]-_v1[1]*_v0[0])/(_v0[0]**2+_v0[1]**2)**0.5
            yaw = math.atan2(_v0[1], _v0[0]) % (2*math.pi)
            break
        long = (point[0]**2+point[1]**2)**0.5
        return dist - 2*long + 0.2*abs(point[2]-yaw)
    
    def cost_obs(self, point, obs_list):
        dist = [((point[0]-obs[0])**2+(point[1]-obs[1])**2)**0.5 for obs in obs_list]
        cost_obs = -self.inf_r
        if len(dist) > 0:
            dist = min(dist)
            if dist <= self.robot_r:
                cost_obs = math.inf
            elif dist <= self.robot_r + self.inf_r:
                cost_obs = self.robot_r-dist
        return 2*cost_obs

    def sample_choice(self, vr, traj, scan):
        choice = []
        for vt in np.arange(vr[0], vr[1]+0.001, self.resol_v):
            vt = round(vt, 2)
            for wt in np.arange(vr[2], vr[3]+0.001, self.resol_w):
                wt = round(wt, 2)
                reach = self.calc_reach(vt, wt)
                cost_v = self.cost_v_w(vt, wt)
                cost_goal = self.cost_goal(reach, traj)
                cost_obs = self.cost_obs(reach, self.scan_to_obs(scan))
                choice.append([vt, wt, cost_v, cost_goal, cost_obs])
        return choice

    def action(self, traj, goal, scan):
        if (len(traj) == 0) or \
           (len(traj) == 1 and goal[0][0]**2+goal[0][1]**2 > (self.last_v*self.dt*2)**2):
            return self.recovery_action()
        traj = np.concatenate([traj, goal], axis=0)
        choice = self.sample_choice(self.calc_dynamic_window(), traj, scan)
        if len(choice) == 0:
            return self.recovery_action()
        score_list = []
        score = np.array(choice)
        for ii in range(0, len(score[:, 0])):
            weights = np.mat([self.v_weight, self.g_weight, self.o_weight])
            scoretemp = weights * (np.mat(score[ii, 2:5])).T
            score_list.append(scoretemp)
        max_score_id = np.argmin(score_list)
        action = list(score[max_score_id, 0:2])
        self.last_v = action[0]
        self.last_w = action[1]
        return action

        
