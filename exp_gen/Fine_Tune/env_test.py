from envs import make_env, read_yaml
from nn import DecisionTransformer, ReturnPred, GPT2Config
from PPOnn import create_net
from pursuit import DWA_Pursuit

import torch, numpy

def check_fail(laser):
    filter_scale = 10
    max_angle = 1.570795
    min_angle = -1.570795
    range_total = laser.size // filter_scale
    angle_step = (max_angle - min_angle) / range_total
    cur_angle = min_angle
    for i in range(range_total):
        if cur_angle >= -0.131 and cur_angle <= 0.131:
            j = i * filter_scale
            if laser[j] <= 0.08:
                return True
        cur_angle += angle_step
    return False

if __name__ == "__main__":
    cfg = read_yaml('envs/cfg/gen_exp.yaml')
    env = make_env(cfg)
    GPT_cfg = GPT2Config(n_embd=cfg['token_dim'], n_layer=cfg['nlayer'], n_head=cfg['nhead'], n_inner=cfg['ninner'], resid_pdrop=cfg['dropout'])
    net = DecisionTransformer(GPT_cfg, None).cuda().float()
    net.load_state_dict(torch.load('model/transformer.pt'))
    net_ReturnPred = ReturnPred(GPT_cfg, None).cuda().float()
    net_ReturnPred.load_state_dict(torch.load('model/return_pred.pt'))
    traj_pursuit = DWA_Pursuit()
    baseline_net = create_net()
    baseline_net.load_state_dict(torch.load('model/baseline.pt'))
    info = {}
    while True:
        traj_pursuit.reset()
        states = env.reset(**info)
        states_seq = []
        for i in range(len(states)):
            states_seq.append([states[i]])
        reward_seq = [net_ReturnPred.pred_return([torch.from_numpy(states[i]).to(dtype=torch.float32, device="cuda") for i in range(len(states))]).cpu().numpy()]
        # reward_seq = [numpy.array([300]).reshape((1, 1))]
        path_seq = []
        timestep_seq = [numpy.array([0])]
        fail = False
        while True:
            if fail == False:
                if check_fail(states[0][0][0]) == True:
                    fail = True
                    continue
                else:
                    path = net.pred_path(
                        [torch.from_numpy(numpy.concatenate(states_seq[i], axis=0)).to(dtype=torch.float32, device="cuda") for i in range(len(states))],
                        torch.from_numpy(numpy.concatenate(reward_seq, axis=0)).to(dtype=torch.float32, device="cuda"), 
                        torch.from_numpy(numpy.concatenate(path_seq, axis=0)).to(dtype=torch.float32, device="cuda") if len(path_seq) != 0 else torch.zeros((0, 2, cfg['path_dim']+1)).to(dtype=torch.float32, device="cuda"), 
                        torch.from_numpy(numpy.concatenate(timestep_seq, axis=0)).to(dtype=torch.int32, device="cuda")
                        ).cpu().numpy()
                    states, reward, done, info = env.step([traj_pursuit.action(numpy.concatenate((path, numpy.array([states[1][0][0], states[1][0][1]]).reshape(1, 2)), axis=0))])
                    if info['all_down'][0]: break
                    if len(timestep_seq) >= 20:
                        for i in range(len(states)):
                            states_seq[i].pop(0)
                        reward_seq.pop(0)
                        path_seq.pop(0)
                        timestep_seq.pop(0)
                    for i in range(len(states)):
                        states_seq[i].append(states[i])
                    return_before = reward_seq[-1]-reward.reshape(reward.shape+(1,))
                    return_now = net_ReturnPred.pred_return([torch.from_numpy(states[i]).to(dtype=torch.float32, device="cuda") for i in range(len(states))]).cpu().numpy()
                    # return_now = return_now+numpy.array([35]).reshape((1, 1)) if return_now[0][0] < 500-35 else return_now
                    for i in range(len(reward_seq)):
                        reward_seq[i] += (return_now - return_before)
                    reward_seq.append(return_now)
                    # reward_seq.append(reward_seq[-1]-reward.reshape(reward.shape+(1,)))
                    path_seq.append(path.reshape((1,)+path.shape))
                    timestep_seq.append(timestep_seq[-1]+numpy.array([1]))
            else:
                pi, _ = baseline_net.forward([torch.tensor(states[i], dtype=torch.float32, device="cuda") for i in range(len(states))], play_mode=True)
                action, _ = pi
                action = action.cpu().detach().numpy()
                states, reward, done, info = env.step(action)
                if info['all_down'][0]: break