import math
import copy

from .utils_op import *

"""
Function to compute Attck_Cost_t based on KLR
"""
def Attack_Cost_Compute(env, init_T, agent_Q, target):
    
    target_policy = DicQ_To_MatrixQ(target, env)
    
    for i in range(len(target_policy)):
        target_policy[i] = (target_policy[i]+0.001)/(1+0.001*env.nA)

    policy = Get_Policy(agent_Q, env)
    T =  env.T.copy()

    # compute P*
    P_star = np.zeros((env.nS*env.nA, env.nS*env.nA))

    for cur_s in range(env.nS):
        for cur_a in range(env.nA):
            row_index = cur_s*env.nA + cur_a
            for next_s in range(env.nS):
                if next_s in target:
                    for next_a in range(env.nA):
                        col_index = next_s*env.nA + next_a
                        P_star[row_index][col_index] = init_T[cur_a][cur_s][next_s]*target_policy[next_s][next_a]
                else:
                    for next_a in range(env.nA):
                        col_index = next_s*env.nA + next_a
                        P_star[row_index][col_index] = init_T[cur_a][cur_s][next_s]*policy[next_s][next_a]


    # compute P_u(s',a'|s,a) with updated policy
    P = np.zeros((env.nS*env.nA, env.nS*env.nA))
    for cur_s in range(env.nS):
        for cur_a in range(env.nA):
            row_index = cur_s*env.nA + cur_a
            for next_s in range(env.nS):
                for next_a in range(env.nA):
                    col_index = next_s*env.nA + next_a
                    P[row_index][col_index] = T[cur_a][cur_s][next_s]*policy[next_s][next_a]


    # compute D_t^KL
    DKL = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            row_index = s*env.nA + a
            for col_index in range(env.nS*env.nA):
                DKL[s][a] += P[row_index][col_index]*math.log(P[row_index][col_index]/P_star[row_index][col_index])
    # stationary distribution          
    w, v = np.linalg.eig(P.T)
    j_stationary = np.argmin(abs(w - 1.0))
    q_stationary = v[:,j_stationary].real
    q_stationary /= q_stationary.sum()

    # compute cost
    cost = 0
    for s in range(env.nS):
        for a in range(env.nA):
            index = s*env.nA + a
            cost += q_stationary[index] * DKL[s][a]

    return cost



"""
Function to measure whether attack done
"""
def Attack_Done_Identify(env, target, Q):
    error = 0
    amount = len(target)

    for s in range(env.nS):
        if s in target:
            target_a_index = np.argmax(target[s])
            learner_a_index = np.argmax(Q[s])

            if target_a_index != learner_a_index:
                error += 1

    if error == 0:
        done = 1
    else:
        done = 0

    accuracy_rate = (amount-error)/amount

    return done, accuracy_rate






if __name__ == "__main__":
    
    # Env
    from envs.env3D_4x4 import GridWorld_3D_env
    env = GridWorld_3D_env()
    
    # Victim
    from victim.victim_Q import VictimAgent_Q
    victim_args = {
        "env": env, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    victim = VictimAgent_Q(**victim_args)
    victim.train(2)
    
    """ target policy """
    target = defaultdict(lambda: np.zeros(env.action_space.n))

    target[0] = np.array([0, 0, 1, 0])
    target[4] = np.array([0, 0, 1, 0])
    target[8] = np.array([0, 0, 1, 0])
    target[12] = np.array([0, 1, 0, 0])
    target[13] = np.array([0, 1, 0, 0])

    print(f"target policy: {target}")
    
    ''' Evaluation '''
    cost = Attack_Cost_Compute(env, victim.init_T, victim.Q, target)
    print(cost)
    
    done, rate = Attack_Done_Identify(env, target, victim.Q)
    print(done, rate)