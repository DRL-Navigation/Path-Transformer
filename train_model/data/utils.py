import random, tf, torch, math
import numpy as np

class TF:
    @classmethod
    def q_to_rpy(cls, q):
        euler = tf.transformations.euler_from_quaternion(q)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        return (roll, pitch, yaw)

    @classmethod
    def rpy_to_q(cls, rpy):
        return tf.transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])

    @classmethod
    def matrix_from_t_q(cls, t, q):
        return tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(t),\
            tf.transformations.quaternion_matrix(q))

    @classmethod
    def matrix_from_t_y(cls, t, y):
        q = TF.rpy_to_q([0, 0, y])
        return TF.matrix_from_t_q(t, q)

    @classmethod
    def matrix_from_t_m(cls, t, m):
        return np.array([[m[0][0], m[0][1], m[0][2], t[0]],
                     [m[1][0], m[1][1], m[1][2], t[1]],
                     [m[2][0], m[2][1], m[2][2], t[2]],
                     [0,       0,       0,       1]
                    ])

    @classmethod
    def matrix_from_pose(cls, pose):
        return TF.matrix_from_t_q(pose[:3], pose[3:])

    @classmethod
    def q_from_matrix(cls, m):
        return tf.transformations.quaternion_from_matrix(m)

    @classmethod
    def t_from_matrix(cls, m):
        return tf.transformations.translation_from_matrix(m)

    @classmethod
    def rpy_from_matrix(cls, m):
        return tf.transformations.euler_from_matrix(m)

    @classmethod
    def inverse(cls, m):
        return tf.transformations.inverse_matrix(m)

    @classmethod
    def transform_point(cls, m, point):
        xyz = tuple(np.dot(m, np.array([point[0], point[1], point[2], 1.0])))[:3]
        return xyz

    @classmethod
    def mul_matrix(cls, m1, m2):
        return np.dot(m1, m2)

def exp_get_batch(exp_pool, batch_size, max_len=3, states_len=3, path_len=3, path_dist=0.8, path_step=0.2):
    states_batch = []
    for _ in range(states_len):
        states_batch.append([])
    reward_batch = []
    path_batch = []
    for _ in range(path_len):
        path_batch.append([])
    mask = []

    batch_i = 0
    while batch_i < batch_size:
        exp = exp_pool.random_get()
        elen = len(exp["reward"])
        if elen < 2: continue
        batch_i += 1
        si = random.randint(0, elen-2)
        exp_cut = exp["states"][si:si+random.randint(1, max_len)]
        tlen = len(exp_cut) if si+len(exp_cut) < elen else len(exp_cut)-1
        for i in range(states_len):
            states_batch[i].append(np.concatenate([exp_cut[j][i] for j in range(tlen)], axis=0))
        reward_batch.append(np.concatenate([exp["reward"][si+j] for j in range(tlen)], axis=0))
        path = []
        for j in range(tlen):
            path_j = get_path(exp_cut[j], exp["states"][si+j:si+j+2*path_len], plen=path_len, pdist=path_dist, pstep=path_step)
            path.append(path_j)
        for i in range(path_len):
            path_batch[i].append(np.concatenate([np.array([path[j][i],]).reshape((1, 1)) for j in range(tlen)], axis=0))

        for i in range(states_len):
            states_batch[i][-1] = np.concatenate([np.zeros((max_len-tlen,)+states_batch[i][-1].shape[1:]), states_batch[i][-1]], axis=0)
        reward_batch[-1] = np.concatenate([np.zeros((max_len-tlen,)+reward_batch[-1].shape[1:]), reward_batch[-1]], axis=0)
        for i in range(path_len):
            path_batch[i][-1] = np.concatenate([np.zeros((max_len-tlen,)+path_batch[i][-1].shape[1:]), path_batch[i][-1]], axis=0)
        mask.append(np.concatenate([np.zeros((1, max_len-tlen)), np.ones((1, tlen))], axis=1))

    for i in range(states_len):
        for j in range(len(states_batch[i])):
            states_batch[i][j] = states_batch[i][j].reshape((1,)+states_batch[i][j].shape)
        states_batch[i] = torch.from_numpy(np.concatenate(states_batch[i], axis=0)).to(dtype=torch.float32, device="cuda")
    for j in range(len(reward_batch)):
        reward_batch[j] = reward_batch[j].reshape((1,)+reward_batch[j].shape)
    reward_batch = torch.from_numpy(np.concatenate(reward_batch, axis=0)).to(dtype=torch.float32, device="cuda")
    for i in range(path_len):
        for j in range(len(path_batch[i])):
            path_batch[i][j] = path_batch[i][j].reshape((1,)+path_batch[i][j].shape)
        path_batch[i] = torch.from_numpy(np.concatenate(path_batch[i], axis=0)).to(dtype=torch.int64, device="cuda")
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.bool, device="cuda")
    # States_batch: [Sensor, Vector, Ped]
    # Size: (Batch, Length, Dim)
    return states_batch, reward_batch, path_batch, mask

def distance(point1, point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5

def intersect(circle, r, point1, point2):
    if point1[0] == point2[0]:
        x = point1[0]
        _delta = (r**2-(x-circle[0])**2)**0.5
        _y1 = circle[1] + _delta
        _y2 = circle[1] - _delta
        y = _y1 if (point1[1]-_y1)*(_y1-point2[1]) >= 0 else _y2
        return (x, y)
    k = (point1[1]-point2[1]) / (point1[0]-point2[0])
    b = point1[1] - k*point1[0]
    _a = k**2 + 1
    _b = 2*k*(b-circle[1])-2*circle[0]
    _c = circle[0]**2+(b-circle[1])**2-r**2
    _delta = (_b**2-4*_a*_c)**0.5
    _x1 = (_delta-_b)/(2*_a)
    _x2 = -(_delta+_b)/(2*_a)
    x = _x1 if (point1[0]-_x1)*(_x1-point2[0]) >= 0 else _x2
    y = k*x+b
    return (x, y)

def get_radian(point1, point2):
    x, y = point1[0]-point2[0], point1[1]-point2[1]
    radian = math.atan2(y, x)
    radian = round(radian/math.pi*180) % 360
    return radian

def get_path(base_state, state_list, plen=5, pdist=1, pstep=0.15):
    path = []
    target_base = TF.matrix_from_t_y((base_state[1][0][0], base_state[1][0][1], 0), base_state[1][0][2])
    target = (base_state[1][0][0], base_state[1][0][1])
    for state in state_list:
        target_new = TF.matrix_from_t_y((state[1][0][0], state[1][0][1], 0), state[1][0][2])
        new_base = TF.mul_matrix(target_base, TF.inverse(target_new))
        point = TF.t_from_matrix(new_base)
        if distance(point, (0, 0)) < pdist:
            path.append((point[0], point[1]))
        else: break

    radian_list = []
    circle = (0, 0)
    pt = 0
    while pt < len(path):
        pt_dist = distance(circle, path[pt])
        if pt_dist > pstep:
            new_circle = intersect(circle, pstep, path[pt], path[pt-1])
            radian = get_radian(new_circle, circle)
            radian_list.append(radian)
            if len(radian_list) >= plen: break
            circle = new_circle
            continue
        pt += 1
    while len(radian_list) < plen:
        radian_list.append(360)

    return radian_list

def make_path_from_radian(radian, path_step):
    path = [[0, 0]]
    for r in radian:
        if r >= 360: break
        ra = r/180*math.pi
        x = path[-1][0] + path_step*math.cos(ra)
        y = path[-1][1] + path_step*math.sin(ra)
        path.append([x, y])
    return np.array(path)

def exp_get_batch_ReturnPred(exp_pool, batch_size, states_len=3):
    states_batch = []
    for _ in range(states_len):
        states_batch.append([])
    reward_batch = []

    for _ in range(batch_size):
        exp = exp_pool.random_get()
        for i in range(len(exp["reward"])):
            for j in range(len(states_batch)):
                states_batch[j].append(exp["states"][i][j])
            reward_batch.append(exp["reward"][i])

    for j in range(len(states_batch)):
        states_batch[j] = torch.from_numpy(np.concatenate(states_batch[j], axis=0)).to(dtype=torch.float32, device="cuda")
    reward_batch = torch.from_numpy(np.concatenate(reward_batch, axis=0)).to(dtype=torch.float32, device="cuda")

    return states_batch, reward_batch