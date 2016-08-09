import numpy as np

def env1(S, A):
    # State Transition
    if A[0, 1] == 1:
        Sn = np.roll(S, 1, axis=1) # Move right
    else:
        Sn = np.roll(S, -1, axis=1) # Move left
    # Reward
    if Sn[0, 0] == 1: # State 0
        R = 1
    elif Sn[0, 9] == 1: # State 9
        R = 100
    else: # State 1~8
        R = 0
    # Terminal or not
    if Sn[0, 0] == 1: # State 0
        T = 1
    elif Sn[0, 9] == 1: # State 9
        T = 1
    else: # State 1~8
        T = 0
    return Sn, R, T

def env2(S, A):
    # State Transitioin
    if A[0, 1] == 1 and S[0, -1] == 0:
        Sn = np.roll(S, 1, axis=1) # Move right
    elif A[0, 1] == 1 and S[0, -1] == 1:
        Sn = S # Stay
    elif A[0, 0] == 1 and S[0, 0] == 0:
        Sn = np.roll(S, -1, axis=1) # Move left
    elif A[0, 0] == 1 and S[0, 0] == 1:
        Sn = S # Stay
    # Reward
    if Sn[0, 0] == 1: # State 0
        R = 1
    elif Sn[0, -1] == 1: # State |S|-1
        R = 100
    else: # Intermediate state
        R = 0
    # Terminal or not
    if Sn[0, 0] == 1: # State 0
        T = 0
    elif Sn[0, 9] == 1: # State |S|-1
        T = 0
    else: # Intermediate state
        T = 0
    return Sn, R, T

