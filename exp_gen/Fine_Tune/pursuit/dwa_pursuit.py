import numpy as np
import math

class DWA_Pursuit:
    def __init__(self):
        self.v_window = [0.0, 0.6]
        self.w_window = [-0.9, 0.9]
        self.acc_v = 5.0
        self.acc_w = 5.0
        self.resol_v = 0.1
        self.resol_w = 0.1
        self.dt = 0.4
        self.v_weight = 1.0
        self.g_weight = 10.0

        self.last_v = 0.0
        self.last_w = 0.0

    def reset(self):
        self.last_v = self.last_w = 0.0

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
        for _ in np.arange(0, self.dt, 0.05):
            reach[0] += vt*math.cos(reach[2])*0.05
            reach[1] += vt*math.sin(reach[2])*0.05
            reach[2] += wt*0.05
        return reach

    def calc_score_v_w(self, vt, wt):
        return abs(vt-self.last_v) + 2*abs(wt-self.last_w)

    def calc_to_goal_cost(self, point, traj):
        for i in range(len(traj)):
            if traj[i][0]**2+traj[i][1]**2 >= 0.25**2:
                traj_points = traj[0:i+1]
                break
            elif i == len(traj)-1:
                traj_points = traj[0:]
        dist_list = [((point[0]-p[0])**2+(point[1]-p[1])**2)**0.5 for p in traj_points]
        min_dist = min(dist_list)
        goal_index = dist_list.index(min_dist)
        if goal_index + 1 >= len(traj):
            goal_index = goal_index - 1
        next_index = goal_index + 1
        goal_yaw = math.atan2(traj[next_index][1]-traj[goal_index][1], traj[next_index][0]-traj[goal_index][0])
        return min_dist + abs(point[2]-goal_yaw)

    def sample_choice(self, vr, traj):
        choice = []
        for vt in np.arange(vr[0], vr[1]+0.001, self.resol_v):
            vt = round(vt, 2)
            for wt in np.arange(vr[2], vr[3]+0.001, self.resol_w):
                wt = round(wt, 2)
                cost_v = self.calc_score_v_w(vt, wt)
                reach = self.calc_reach(vt, wt)
                cost_dis2goal = self.calc_to_goal_cost(reach, traj)
                choice.append([vt, wt, cost_v, cost_dis2goal])
        return choice

    def action(self, traj):
        if len(traj) == 0:
            return [0.0, 0.0]
        vr = self.calc_dynamic_window()
        choice = self.sample_choice(vr, traj)
        if len(choice) == 0:
            return [0.0, 0.0]
        score_list = []
        score = np.array(choice)
        for ii in range(0, len(score[:, 0])):
            weights = np.mat([self.v_weight, self.g_weight])
            scoretemp = weights * (np.mat(score[ii, 2:4])).T
            score_list.append(scoretemp)
        max_score_id = np.argmin(score_list)
        action = list(score[max_score_id, 0:2])
        self.last_v = action[0]
        self.last_w = action[1]
        return action

        
