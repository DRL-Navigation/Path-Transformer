import logging
import gym
import numpy as np
import math
import yaml
import time, sys

from typing import *
from collections import deque
from copy import deepcopy


from envs.state import ImageState
from envs.action import *
from envs.utils import BagRecorder


class StatePedVectorWrapper(gym.ObservationWrapper):
    avg = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0])
    std = np.array([6.0, 6.0, 0.6, 0.9, 0.50, 0.5, 6.0])

    def __init__(self, env, cfg=None):
        super(StatePedVectorWrapper, self).__init__(env)

    def observation(self, state: ImageState):
        self._normalize_ped_state(state.ped_vector_states)
        return state

    def _normalize_ped_state(self, peds):

        for robot_i_peds in peds:
            for j in range(int(robot_i_peds[0])): # j: ped index
                robot_i_peds[1 + j * 7:1 + (j + 1) * 7] = (robot_i_peds[1 + j * 7:1 + (j + 1) * 7] - self.avg) / self.std


class VelActionWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(VelActionWrapper, self).__init__(env)
        if cfg['discrete_action']:
            self.actions: DiscreteActions = DiscreteActions(cfg['discrete_actions'])

            self.f = lambda x: self.actions[int(x)] if np.isscalar(x) else ContinuousAction(*x)
        else:
            clip_range = cfg['continuous_actions']

            def tmp_f(x):
                y = []
                for i in range(len(x)):
                    y.append(np.clip(x[i], clip_range[i][0], clip_range[i][1]))
                return ContinuousAction(*y)
            # self.f = lambda x: ContinuousAction(*x)
            self.f = tmp_f

    def step(self, action: np.ndarray, **kwargs):
        action = self.action(action)
        state, reward, done, info = self.env.step(action, **kwargs)
        info['speeds'] = np.array([a.reverse()[:2] for a in action])
        return state, reward, done, info

    def action(self, actions: np.ndarray) -> List[ContinuousAction]:
        return list(map(self.f, actions))

    def reverse_action(self, actions):

        return actions


class MultiRobotCleanWrapper(gym.Wrapper):
    is_clean : list
    def __init__(self, env, cfg):
        super(MultiRobotCleanWrapper, self).__init__(env)
        self.is_clean = np.array([True] * cfg['agent_num_per_env'])

    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        info['is_clean'] = deepcopy(self.is_clean)
        reward[~info['is_clean']] = 0
        info['speeds'][~info['is_clean']] = np.zeros(2)
        # for i in range(len(done)):
        #     if done[i]:
        #         self.is_clean[i]=False
        self.is_clean = np.where(done>0, False, self.is_clean)
        return state, reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.is_clean = np.array([True] * len(self.is_clean))
        return state



class StateBatchWrapper(gym.Wrapper):
    batch_dict: np.ndarray

    def __init__(self, env, cfg):
        print(cfg,flush=True)
        super(StateBatchWrapper, self).__init__(env)
        self.q_sensor_maps = deque([], maxlen=cfg['image_batch']) if cfg['image_batch']>0 else None
        self.q_vector_states = deque([], maxlen=cfg['state_batch']) if cfg['state_batch']>0 else None
        self.q_lasers = deque([], maxlen=cfg['laser_batch']) if cfg['laser_batch']>0 else None
        self.batch_dict = {
            "sensor_maps": self.q_sensor_maps,
            "vector_states": self.q_vector_states,
            "lasers": self.q_lasers,
        }

    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        return self.batch_state(state), reward, done, info

    def _concate(self, b: str, t: np.ndarray):
        q = self.batch_dict[b]
        if q is None:
            return t
        else:
            t = np.expand_dims(t, axis=1)
        # start situation
        while len(q) < q.maxlen:
            q.append(np.zeros_like(t))
        q.append(t)
        #  [n(Robot), k(batch), 84, 84]
        return np.concatenate(list(q), axis=1)

    def batch_state(self, state):
        # TODO transpose. print
        state.sensor_maps = self._concate("sensor_maps", state.sensor_maps)
        # print('sensor_maps shape; ', state.sensor_maps.shape)

        # [n(robot), k(batch), state_dim] -> [n(robot), k(batch) * state_dim]
        tmp_ = self._concate("vector_states", state.vector_states)
        state.vector_states = tmp_.reshape(tmp_.shape[0], tmp_.shape[1] * tmp_.shape[2])
        # print("vector_states shape", state.vector_states.shape)
        state.lasers = self._concate("lasers", state.lasers)[None]
        # print("lasers shape:", state.lasers.shape)
        return state

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return self.batch_state(state)


class SensorsPaperRewardWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(SensorsPaperRewardWrapper, self).__init__(env)

        self.ped_safety_space = cfg['ped_safety_space']
        self.cfg = cfg

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        return states, self.reward(reward, states), done, info

    def _each_r(self, states: ImageState, index: int):
        distance_reward_factor = 200
        collision_reward = reach_reward = step_reward = distance_reward = rotation_reward = beep_reward = 0

        min_dist = states.ped_min_dists[index]
        vector_state = states.vector_states[index]
        is_collision = states.is_collisions[index]
        is_arrive = states.is_arrives[index]
        step_d = states.step_ds[index]

        if is_collision > 0:
            collision_reward = -500
        else:
            d = math.sqrt(vector_state[0] ** 2 + vector_state[1] ** 2)
            # print 'robot ',i," dist to goal: ", d
            if d < 0.3 or is_arrive:
                reach_reward = 500.0
            else:
                distance_reward = (step_d * distance_reward_factor) // 1
                step_reward = -50

        # reward = collision_reward + reach_reward + step_reward + distance_reward + beep_reward
        reward = reach_reward + step_reward + collision_reward + distance_reward
        return reward

    def reward(self, reward, states):
        rewards = np.zeros(len(states))
        for i in range(len(states)):
            rewards[i] = self._each_r(states, i)

        return rewards


class NeverStopWrapper(gym.Wrapper):
    """
        NOTE !!!!!!!!!!!
        put this in last wrapper.
    """
    def __init__(self, env, cfg):
        super(NeverStopWrapper, self).__init__(env)

    def step(self, action):
        states, reward, done, info = self.env.step(action)
        if info['all_down'][0]:
            states = self.env.reset(**info)

        return states, reward, done, info


# time limit
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeLimitWrapper, self).__init__(env)
        self._max_episode_steps = cfg['time_max']
        robot_total = cfg['robot']['total']
        self._elapsed_steps = np.zeros(robot_total, dtype=np.uint8)

    def step(self, ac, **kwargs):
        observation, reward, done, info = self.env.step(ac, **kwargs)
        self._elapsed_steps += 1
        done = np.where(self._elapsed_steps > self._max_episode_steps, 1, done)
        info['dones_info'] = np.where(self._elapsed_steps > self._max_episode_steps, 10, info['dones_info'])
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)###


class InfoLogWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(InfoLogWrapper, self).__init__(env)
        self.robot_total = cfg['robot']['total']
        self.tmp = np.zeros(self.robot_total, dtype=np.uint8)
        self.ped: bool = cfg['ped_sim']['total'] > 0 and cfg['env_type'] == 'robot_nav'

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        info['arrive'] = states.is_arrives
        info['collision'] = states.is_collisions

        info['dones_info'] = np.where(states.is_collisions > 0, states.is_collisions, info['dones_info'])
        info['dones_info'] = np.where(states.is_arrives == 1, 5, info['dones_info'])
        info['all_down'] = self.tmp + sum(np.where(done>0, 1, 0)) == len(done)

        if self.ped:
            # when robot get close to human, record their speeds.
            info['bool_get_close_to_human'] = np.where(states.ped_min_dists < 1 , 1 , 0)

        return states, reward, done, info


class BagRecordWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(BagRecordWrapper, self).__init__(env)
        self.bag_recorder = BagRecorder(cfg["bag_record_output_name"])
        self.record_epochs = int(cfg['bag_record_epochs'])
        self.episode_res_topic = "/" + cfg['env_name'] + str(cfg['node_id']) + "/episode_res"
        print("epi_res_topic", self.episode_res_topic, flush=True)
        self.cur_record_epoch = 0

        self.bag_recorder.record(self.episode_res_topic)

    def _trans2string(self, dones_info):
        o: List[str] = []
        for int_done in dones_info:
            if int_done == 10:
                o.append("stuck")
            elif int_done == 5:
                o.append("arrive")
            elif 0 < int_done < 4:
                o.append("collision")
            else:
                raise ValueError
        print(o, flush=True)
        return o

    def reset(self, **kwargs):
        if self.cur_record_epoch == self.record_epochs:
            time.sleep(10)
            self.bag_recorder.stop()
        if kwargs.get('dones_info') is not None: # first reset not need
            self.env.end_ep(self._trans2string(kwargs['dones_info']))
            self.cur_record_epoch += 1
        """
                done info:
                10: timeout
                5:arrive
                1: get collision with static obstacle
                2: get collision with ped
                3: get collision with other robot
                """
        print(self.cur_record_epoch, flush=True)
        return self.env.reset(**kwargs)


class TrajectoryPathHelper():
    """
    statistics for path
    """

    v_array = []
    w_array = []

    "path terms"
    v_acc = 0
    v_jerk = 0 # average jerk (time derivative of acceleration) of the robot over its entire trajectory
    w_acc = 0
    w_jerk = 0
    w_zero = 0 # the times of w changes its sign. (+ -> - || - > +)
    w_variance = 0 # variance of w

    def __init__(self, dt):
        self.dt = dt

    def add_vw(self, v, w):
        self.v_array.append(v)
        self.w_array.append(w)

    def get_w_zero(self):
        return self.w_zero

    def get_w_variance(self):
        return self.w_variance
    
    def get_w_acc(self):
        return self.w_acc

    def get_w_jerk(self):
        return self.w_jerk
    
    def get_v_acc(self):
        return self.v_acc

    def get_v_jerk(self):
        return self.v_jerk

    def cal_w_variance(self):
        self.w_variance = np.var(self.w_array)
        return self.w_variance

    def cal_w_zero(self):
        tmp = 0
        w_zero = 0
        for w in self.w_array:
            if w == 0:
                if tmp != 0:
                    w_zero += 1
            else:
                if (w>0 and tmp < 0) or (w<0 and tmp>0):
                    w_zero += 1
            tmp = w
        self.w_zero = w_zero
        return w_zero

    def cal_jerk_acc(self):
        dt = self.dt
        # print(self.w_array)
        # v_jrk
        v_acc = np.diff(self.v_array, axis=0) / dt
        self.v_acc = np.average(np.abs(v_acc))
        v_jrk = np.diff(v_acc, axis=0) / dt
        self.v_jerk = np.average(np.abs(v_jrk))  #  np.average(np.square(v_jrk))
        # w_jerk
        w_acc = np.diff(self.w_array, axis=0) / dt
        self.w_acc = np.average(np.abs(w_acc))
        w_jrk = np.diff(w_acc, axis=0) / dt
        self.w_jerk = np.average(np.abs(w_jrk))  #  np.average(np.square(w_jrk))
        # print('w_jerk',self.w_jerk)
        return self.v_jerk, self.w_jerk, self.v_acc, self.w_acc

    def clear_vw_array(self):
        self.v_array.clear()
        self.w_array.clear()
        self.w_zero = 0
        self.w_variance = 0
        self.w_jerk = 0
        self.v_jerk = 0
        self.w_acc = 0
        self.v_acc = 0

    def reset(self):
        self.cal_w_variance()
        self.cal_w_zero()
        self.cal_jerk_acc()

        return


class TestEpisodeWrapper(gym.Wrapper):
    """

    for one robot
    """
    def __init__(self, env, cfg):
        super(TestEpisodeWrapper, self).__init__(env)
        self.cur_episode = 0
        self.max_episodes = cfg["init_pose_bag_episodes"]
        self.dt = cfg['control_hz']
        print("[TestEpisodeWrapper]: Concurrent dt:", self.dt, flush=True)
        self.arrive_num = 0
        self.static_coll_num = 0
        self.ped_coll_num = 0
        self.other_coll_num = 0
        self.traj_coll_num = 0
        self.steps = 0
        self.tmp_steps = 0
        self.stuck_num = 0
        self.v_sum = 0
        self.w_sum = 0
        self.speed_step = 0
        self.all_step = 0

        self.w_variance_array = []
        self.v_jerk_array = []
        self.w_jerk_array = []
        self.w_zero_array = []
        self.v_acc_array = []
        self.w_acc_array = []

        self.traj_helper = TrajectoryPathHelper(dt=self.dt)

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        self.traj_coll_num += 1 if states.traj_collisions[0] != 0 else 0
        self.tmp_steps += 1
        self.all_step += 1
        speeds = info.get("speeds")[0] # suppose only one agent here
        self.v_sum += speeds[0]
        self.w_sum += abs(speeds[1])

        self.traj_helper.add_vw(*speeds[:2])
        return states, reward, done, info

    def reset(self, **kwargs):
        if self.tmp_steps > 3: # 就2 3步，太短的一般来说有问题
            self.cur_episode += 1
            self.dones_statistics(kwargs.get('dones_info'))

        if self.cur_episode == self.max_episodes:
            self.screen_out()
        self.tmp_steps = 0
        return self.env.reset(**kwargs)

    def dones_statistics(self, t):
        if t is not None:
            t = t[0]
            self.speed_step += self.tmp_steps
            if t == 5:
                self.arrive_num += 1
                self.steps += self.tmp_steps
            elif t == 10:
                self.stuck_num += 1
            elif t == 1:
                self.static_coll_num += 1
            elif t == 2:
                self.ped_coll_num += 1
            elif t == 3:
                self.other_coll_num += 1
            else:
                print("[TestEpisodeWrapper]: No dones info: ", t)
                raise ValueError

            self.path_statistics()

    def path_statistics(self):
        self.traj_helper.reset()
        self.v_jerk_array.append(self.traj_helper.get_v_jerk())
        self.w_jerk_array.append(self.traj_helper.get_w_jerk())
        self.w_zero_array.append(self.traj_helper.get_w_zero())
        self.w_variance_array.append(self.traj_helper.get_w_variance())
        self.v_acc_array.append(self.traj_helper.get_v_acc())
        self.w_acc_array.append(self.traj_helper.get_w_acc())

        self.traj_helper.clear_vw_array()

    def screen_out(self):
        print("""\n
                    ###############################
                    [TestEpisodeWrapper]: Have run max episodes {}, statistics number are in the following:

                    arrive_rate: {},
                    static_coll_rate: {},
                    ped_coll_rate: {},
                    other_coll_rate: {},
                    stuck_rate: {},
                    avg_arrive_steps: {},
                    path_coll_rate: {},
                    avg_v: {},
                    avg_w: {},
                    avg_v_acc: {},
                    avg_w_acc: {},
                    avg_v_jerk: {},
                    avg_w_jerk: {},
                    avg_w_zero: {},
                    avg_w_variance: {},
                    """.format(self.max_episodes,
                               self.arrive_num / self.max_episodes,
                               self.static_coll_num / self.max_episodes,
                               self.ped_coll_num / self.max_episodes,
                               self.other_coll_num / self.max_episodes,
                               self.stuck_num / self.max_episodes,
                               self.steps / max(1, self.arrive_num),
                               self.traj_coll_num / self.all_step,
                               self.v_sum / self.speed_step,
                               self.w_sum / self.speed_step,
                               sum(self.v_acc_array) / self.max_episodes,
                               sum(self.w_acc_array) / self.max_episodes,
                               sum(self.v_jerk_array) / self.max_episodes,
                               sum(self.w_jerk_array) / self.max_episodes,
                               sum(self.w_zero_array) / self.max_episodes,
                               sum(self.w_variance_array) / self.max_episodes,
                               ), flush=True)
        print("[TestEpisodeWrapper]: Exit Progress!")
        sys.exit()



