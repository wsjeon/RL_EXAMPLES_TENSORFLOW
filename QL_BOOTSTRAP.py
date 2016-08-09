# ============================================================================
# Wonseok Jeon, EE, KAIST
# 2016/08/08: Q Learning with Randomized Value Functions
# Reference: Deep Exploration via Bootstrapped DQN
# ============================================================================
import tensorflow as tf
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt
from environments import env1, env2


# ============================================================================
# Graph Construction
# ============================================================================
# (0) Parameters
gamma = 0.9 # discount factor
alpha = 0.2 # learning rate
mem_size = 1 # experience memory size
batch_size = 1 # batch size < experience memory size
update_counter = 1 # update Q for time step
K = 10 # number of Q headers
prob = 1  # probability to update each Q
replay = 0 # replay or not
STDDEV = 10 # standard deviation for randomization

# (1) Q values
W = tf.Variable(tf.random_normal([10, 2*K],stddev=STDDEV), name="W")
W_ = tf.Variable(tf.random_normal([10, 2*K],stddev=STDDEV), name="W_")
WeightCopy = W_.assign(W) # Weight Copying op. for Q learning

# (2) State, Action, Reward, Next State, Mask
S = tf.placeholder("float", shape=[None, 10], name="State") # ?x10
A = tf.placeholder("float", shape=[None, 2*K], name="Action") # ?x2K
R = tf.placeholder("float", shape=[None, 1], name="Reward") # ?x1
S_ = tf.placeholder("float", shape=[None, 10], name="Next_State") #?x10
M = tf.placeholder("float", shape=[None, 2*K], name="Bootstrap_Mask") #?x2K

# (3) Resultant Q for given state
Q = tf.matmul(S, W) # ?x2K
Q_ = tf.matmul(S_, W_) # ?x2K

# (4) Objective function
Y = tf.add(R, tf.mul(gamma, tf.mul(Q_, M))) # ?x1
Y_ = tf.placeholder("float", shape=[None, 2*K], name="TargetQ") # ?x1
cost = tf.reduce_sum(tf.square(tf.sub(Y_, tf.mul(A, tf.mul(Q, M))))) # 1x1
init = tf.initialize_all_variables() # initializer
optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost) # optimizer

# (5) Launch the graph
sess = tf.InteractiveSession() # Launch the graph in a session -> default
tf.get_default_graph().finalize() # Read-only graph 

# ============================================================================
# Agent's tool
# ============================================================================
memory = np.zeros([4, mem_size]) # S, A, R, S': memory
Mmemory = np.zeros([K, mem_size]) # M: mask memory
Stable = np.identity(10) # State representation table
Atable = np.identity(2) # Action representation table
Mtable = np.zeros([K, 2*K]) # Mask representation table
Mtable[:,0:(2*K):2] = np.identity(K)
Mtable[:,1:(2*K):2] = np.identity(K)

Tot_Reward = np.zeros([20])

for tot_episode in range(10,201,10): 
    for epoch in range(1,101,1): # 
        sess.run(init) # initialization
    	with tf.device("/gpu:0"):
            # train
            counter = 1
            tot_time_step = 1
      
            for episode in range(tot_episode + 1): # Do multiple episodes
               	start_time = time.time()
                epsilon = 1 - float(episode) / float(tot_episode)
                St = Stable[[1], :] # initialize current state
                T = 0 # initialize T
                SelectQ = np.random.randint(K) # Choose Q to take action
                time_step = 1
                while(not T):
                    # 1. Check current Q value.
                    Qt = sess.run(Q, feed_dict={S: St}) # 1x2K
                    Qt = Qt * Mtable[[SelectQ], :]
                    rows, cols = np.where(Mtable[[SelectQ], :]) # find mask position
                    
                    # 2. Select action using epsilon greedy policy 
                    if np.random.uniform()>epsilon:
                        if Qt[0, cols[0]] > Qt[0, cols[1]]:
                            At = 0
                        elif Qt[0, cols[0]] < Qt[0, cols[1]]:
                            At = 1
                        else:
                            At = np.random.randint(0, 2)
                    else:
                        At = np.random.randint(0, 2)
                    if At == 0:
                        At = np.array([[1, 0]])
                    else:
                        At = np.array([[0, 1]])
                    # 3. Do action & Get next state and reward.
                    Sn, Rn, T = env2(St, At) # interaction with Environment

                    # 4. Update the memory
                    memory = np.roll(memory, 1, axis=1) 
                    memory[0, 0] = np.argmax(St)
                    memory[1, 0] = np.argmax(At)
                    memory[2, 0] = Rn
                    memory[3, 0] = np.argmax(Sn)

                    Mmemory = np.roll(Mmemory, 1, axis=1)
                    Mmemory[:, [0]] = (np.random.uniform(size=[K, 1]) < prob).astype(int)
                
                    # 5. Update Q value by using the memory
                    if counter == update_counter:
                        # Select reward and state from the memory
                        if replay == 1:
                            Sampling = np.random.randint(min(tot_time_step, mem_size), size=(batch_size))
                        else:
                            Sampling = range(batch_size)
                        Sindices = memory[0, Sampling].astype(int).tolist()
                        Supdate = Stable[Sindices, :]
                        Aindices = memory[1, Sampling].astype(int).tolist()
                        Aupdate = np.kron(np.ones([1, K]), Atable[Aindices, :])
                        Rupdate = memory[[2], [Sampling]].T
                        S_indices = memory[3, Sampling].astype(int).tolist()
                        S_update = Stable[S_indices, :]
                        Mupdate = np.matmul(Mmemory[:, Sampling ].T, Mtable)
                        Y_target = Y.eval({R: Rupdate, S_: S_update, M: Mupdate})
                        # Update Parameters (Q values)                
                        sess.run(optimizer, feed_dict={S: Supdate, A: Aupdate, Y_:
                            Y_target, M: Mupdate})
                        # Reset counter
                        counter = 1
                        # Weight Copying
                        sess.run(WeightCopy)

                    else:
                        # Add 1 to counter
                        counter = counter +1
                    # 6. State Transition
                    Stmp = St
                    St = Sn
                    os.system('clear')                
                    time_step = time_step + 1 
                    tot_time_step = tot_time_step + 1

		    # 7. Print the status
                    print "============================================================="
                    print "          Q Learning via Epsilon Greedy Policy"
                    print "                 (Q", cols[0]/2, "is currently used)"
                    print "   Annealing:",tot_episode,", Epoch:",epoch,", Episode:",episode,", Time Step:",time_step
                    print (time.time() - start_time), "seconds"
                    print "============================================================="

                    print "curr State :", Stmp
                    if At[0, 0] == 1:
                        print "Action: Left"
                    else:
                        print "Action: Right"
                    print "Reward:", Rn
                    print "next State:", Sn
                    print "Q values:"
                    print sess.run(W)
                    print "Trained reward with %d-episode annealing: %d" % (tot_episode, Tot_Reward[tot_episode/10-1])
                    print "Epsilon:", epsilon
                    if (Rn == 100):
                        print "============================================================="
                        print "                         100 Reward "
                        print "============================================================="
                    if (time_step > 19):
                        break
            Tot_Reward[tot_episode/10-1] = Tot_Reward[tot_episode/10-1] + Rn

            # Save data and plot
    Episode_Index = np.arange(10, 201, 10)
    np.savetxt('Tot_Reward_EPS_STDDEV_'+str(STDDEV)+'.txt', Tot_Reward) 
    plt.plot(Episode_Index, Tot_Reward/100, 'bs')
    plt.xlabel('Num. of Annealing Episodes')
    plt.ylabel('Average Reward over 100 Run')
    plt.savefig('Avg_Reward_EPS_STDDEV_'+str(STDDEV)+'.png')


sess.close()            





