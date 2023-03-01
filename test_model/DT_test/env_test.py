from envs import make_env, read_yaml
from nn import DecisionTransformer, ReturnPred, GPT2Config
from pursuit import DWA_Pursuit, Pure_Pursuit
from data import make_path_from_radian

import torch, numpy

if __name__ == "__main__":
    cfg = read_yaml('envs/cfg/circle.yaml')
    max_len = cfg['max_len']
    env = make_env(cfg)
    GPT_cfg = GPT2Config(n_embd=cfg['token_dim'], n_layer=cfg['nlayer'], n_head=cfg['nhead'], n_inner=cfg['ninner'], resid_pdrop=cfg['dropout'])
    net = DecisionTransformer(GPT_cfg).cuda().float()
    net.load_state_dict(torch.load('model/transformer.pt'))
    GPT_cfg = GPT2Config(n_embd=512)
    net_ReturnPred = ReturnPred(GPT_cfg).cuda().float()
    net_ReturnPred.load_state_dict(torch.load('model/return_pred.pt'))
    traj_pursuit = DWA_Pursuit()
    # traj_pursuit = Pure_Pursuit()
    info = {}
    while True:
        traj_pursuit.reset()
        states = env.reset(**info)
        states_seq = []
        for i in range(len(states)):
            states_seq.append([states[i]])
        reward_seq = [net_ReturnPred.pred_return([torch.from_numpy(states[i]).to(dtype=torch.float32, device="cuda") for i in range(len(states))]).to(dtype=torch.int64, device="cpu").to(dtype=torch.float32).numpy()]
        # reward_seq = [numpy.array([450]).reshape((1, 1))]
        radian_seq = []
        for _ in range(cfg['path_dim']):
            radian_seq.append([])
        while True:
            states_torch = [torch.from_numpy(numpy.concatenate(states_seq[i], axis=0)).to(dtype=torch.float32, device="cuda") for i in range(len(states_seq))]
            reward_torch = torch.from_numpy(numpy.concatenate(reward_seq, axis=0)).to(dtype=torch.float32, device="cuda")
            radian_torch = [torch.from_numpy(numpy.concatenate(radian_seq[i]+[numpy.zeros((1, 1)),], axis=0)).to(dtype=torch.int64, device="cuda") for i in range(len(radian_seq))]
            radian = net.pred_path(states_torch, reward_torch, radian_torch)
            path = make_path_from_radian(radian, cfg['path_step'])
            goal = numpy.array([states[1][0][0], states[1][0][1]]).reshape(1, 2)
            scan = states[0].reshape(-1)
            action = traj_pursuit.action(path, goal, scan)
            # print(action, flush=True)
            states, reward, done, info = env.step([action,], path=path.reshape((1,)+path.shape))
            if info['all_down'][0]: break
            if len(path) > 1:
                if len(reward_seq) >= max_len:
                    for i in range(len(states)):
                        states_seq[i].pop(0)
                    reward_seq.pop(0)
                    for i in range(cfg['path_dim']):
                        radian_seq[i].pop(0)
                for i in range(len(states)):
                    states_seq[i].append(states[i])
                return_before = reward_seq[-1]-reward.reshape(reward.shape+(1,))
                return_now = net_ReturnPred.pred_return([torch.from_numpy(states[i]).to(dtype=torch.float32, device="cuda") for i in range(len(states))]).to(dtype=torch.int64, device="cpu").to(dtype=torch.float32).numpy()
                if return_before[0][0] > return_now[0][0]:
                    for i in range(len(reward_seq)):
                        reward_seq[i] += (return_now - return_before)
                    reward_seq.append(return_now)
                else: reward_seq.append(return_before)
                # reward_seq.append(reward_seq[-1]-reward.reshape(reward.shape+(1,)))
                for i in range(cfg['path_dim']):
                    radian_seq[i].append(numpy.array([radian[i],]).reshape(1, 1))
            else:
                states_seq = []
                for i in range(len(states)):
                    states_seq.append([states[i]])
                reward_seq = [net_ReturnPred.pred_return([torch.from_numpy(states[i]).to(dtype=torch.float32, device="cuda") for i in range(len(states))]).to(dtype=torch.int64, device="cpu").to(dtype=torch.float32).numpy()]
                # reward_seq = [numpy.array([450]).reshape((1, 1))]
                radian_seq = []
                for _ in range(cfg['path_dim']):
                    radian_seq.append([])