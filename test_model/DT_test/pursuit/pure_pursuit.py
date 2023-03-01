import numpy as np
import math

class Pure_Pursuit:
    def __init__(self):
        self.v = 0.5

    def reset(self):
        pass

    def recovery_action(self):
        return [0.0, 0.9]
    
    def action(self, traj, goal, *arg):
        if (len(traj) == 0) or \
           (len(traj) == 1 and goal[0][0]**2+goal[0][1]**2 > 0.2**2):
            return self.recovery_action()
        traj = np.concatenate([traj, goal], axis=0)
        return [self.v, 2*traj[1][1]*self.v/(traj[1][0]**2+traj[1][1]**2)]