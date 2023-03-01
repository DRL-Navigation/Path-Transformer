import sys
sys.path.append(sys.path[0] + "/../../")
import random
import yaml
import time

from dwa_utils import *

from envs import make_env, read_yaml


class DWAPolicy4Nav:
    def __init__(self, n, cfg):
        self.n = n
        self.vw = [(0, 0)] * n
        self.config_env = cfg

    def gen_action(self, laser, goal):
        out = []

        for i in range(self.n):
            vw  = choose_dwa_action(laser[i][0], goal[i], self.vw[i], self.config_env['view_angle_begin'],
                              self.config_env['view_angle_end'], self.config_env['laser_max'])[0]
            # print(vw, flush=True)
            self.vw[i] = tuple(vw[:2])
            out.append( (vw[0], vw[1], random.uniform(-0.6, 0.6) ) )

        # print(out)
        return out


if __name__ == "__main__":
    import sys
    tmp = len(sys.argv)
    if tmp == 2:
        cfg = read_yaml(sys.argv[1])
    else:
        cfg = read_yaml('envs/cfg/gen_exp.yaml')
    env = make_env(cfg)
    # env2 = make_env(cfg)
    # time.sleep(1)
    dwa_policy = DWAPolicy4Nav(env.robot_total, cfg)
    # test continuous action

    state = env.reset()
    info = {'sub_goal': state[1]}
    while True:
        # a = time.time()
        state, reward, done, info = env.step(dwa_policy.gen_action(state[0], info['sub_goal']))
        # env2.step(random_policy.gen_action())
        # print(time.time() - a)
        # time.sleep(0.1)

