
import numpy as np
import time
import sys
import tkinter as tk
import numpy as np
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)
import time
import random


'''HYPERPARAMETERS'''

#steps = 30000                                      # number of learning steps
start_increase_e_greedy = 10000          # start increse greedy action
k_reward = 10                                      # reward multiplication
'''net'''
bias_on = True                           # on\off  bias (True\False)
n_n = 7                                 # n_neurons_hidden_layer
l_r = 0.001                             # learning_rate
r_w = 0.9                              # reward_decay
e_g_enc = 0.001                     # e_greedy_ecrement
r_t_i = 1                            # replace_target_iter
m_s = 5000                           # memory_size
b_s = 32                             # mini_batch_size
epsil = 0.9                            # start greedy-coefficient                     
    

''' enviroment'''
n_f = 1                                            # count of features: power1, power2, "frequency"........ 
random_consumption_on = True       #random consumptiot in timeline  no\off  (True\False)

# start parameters:
p_1 = 200
p_2 = 100
c = 300
k_power=10        #  normalisation input power
k_freq=1            #  normalisation input "freq"

max_p_1 = 220                                # max power 1 station
min_p_1  = 180                                # min power 1 station

max_p_2 = 110                                # max power 1 station
min_p_2  = 90                                 # min power 1 station

max_c  = 310                              # max power consumption
min_c   = 290                                 # min power 1 station

#act = ['up_power1', 'down_power1', 'up_power2', 'down_power2', 'no actions']         # actions
#act = ['up_power1', 'down_power1']                                                                       # actions
act = ['up_power1', 'down_power1', 'up_power2', 'down_power2']  

output_graph = True                  # make graph



class Sys(tk.Tk, object):
    def __init__(self):
        super(Sys, self).__init__()
        self.action_space = act
        self.n_actions = len(self.action_space)
        self.n_features = n_f   
        self.title('POWER SYSTEM')
        self.geometry('{0}x{1}'.format(500, 500))   # dimentions
        self.consumption = c
        self.val_1power = p_1
        self.val_2power = p_2
        self.frequency = 50 
        self._build_system()         
      
        
    def _build_system(self):
        self.canvas = tk.Canvas(self, bg='lightgreen',  height=500, width=500)   # dimentions
        ''' power system '''
        self.oval = self.canvas.create_oval(50,50,100,100, fill = "gray",outline = 'black')    # 1 Power station
        self.oval = self.canvas.create_arc([60,70], [75, 85], extent = 180, style = "arc")  
        self.oval = self.canvas.create_arc([75,70], [90, 85], extent = -180, style = "arc")
        self.oval = self.canvas.create_oval(50,350,100,400, fill = "gray", outline = 'black')    # 2 power station
        self.arc = self.canvas.create_arc([60,370], [75, 385], extent = 180, style = "arc") 
        self.arc = self.canvas.create_arc([75,370], [90, 385], extent = -180, style = "arc")
        self.line = self.canvas.create_line(100,75, 300, 75, fill = 'black')
        self.line = self.canvas.create_line(100,375, 300, 375, fill = 'black')
        self.line = self.canvas.create_line(300,375, 300, 75, fill = 'black')
        self.line = self.canvas.create_line(275,225, 350, 225, fill = 'black')
        self.oval = self.canvas.create_line(340,225, 340, 250, fill = 'black')
        self.oval = self.canvas.create_polygon([335,250],[340,260],[345,250], fill="black", outline = "black")
        

        '''POWER STATION : '''
        self.power1 = self.canvas.create_text(50,120, text = str(self.val_1power))
        self.power2 = self.canvas.create_text(50,420, text = str(self.val_2power))
        '''CONSUMPTION : '''
        self.cons = self.canvas.create_text(380,250, text = str(self.consumption))
        '''FREQUENCY'''
        self.freq_text = self.canvas.create_text(200,20, text = str('FREQUENCY ='),fill="black")
        self.freq = self.canvas.create_text(300,20, text = str(self.frequency),fill="red")
        
        #debug:
        '''button'''
        '''CONSUMPTION'''
        self.but = tk.Button( text = "UP")
        # Call on function change_label with amount = 10
        self.but.bind("<Button-1>", lambda event: self.increase_consumption())
        self.but.place(relx=0.75, rely = 0.6, anchor = "center")
        self.but = tk.Button(text = "DOWN")
        # Call on function change_label with amount = -10
        self.but.bind("<Button-1>", lambda event: self.decrease_consumption())
        self.but.place(relx=0.85, rely = 0.6, anchor = "center")
        #GENERATION:
        self.but = tk.Button( text = "1 UP")
        # Call on function change_label with amount = 10
        self.but.bind("<Button-1>", lambda event: self.increase_1power())
        self.but.place(relx=0.2, rely = 0.25, anchor = "center")
        self.but = tk.Button( text = "1 DOWN")
        # Call on function change_label with amount = 10
        self.but.bind("<Button-1>", lambda event: self.decrease_1power())
        self.but.place(relx=0.3, rely = 0.25, anchor = "center")
        self.but = tk.Button( text = "2 UP")
        # Call on function change_label with amount = 10
        self.but.bind("<Button-1>", lambda event: self.increase_2power())
        self.but.place(relx=0.2, rely = 0.85, anchor = "center")
        self.but = tk.Button( text = "2 DOWN")
        # Call on function change_label with amount = 10
        self.but.bind("<Button-1>", lambda event: self.decrease_2power())
        self.but.place(relx=0.3, rely = 0.85, anchor = "center")
        '''
        #RESET
        self.but = tk.Button( text = "RESET")
        # Call on function change_label with amount = 10
        self.but.bind("<Button-1>", lambda event: self.reset())
        self.but.place(relx=0.5, rely = 0.5, anchor = "center")
        '''
        self.canvas.pack()  # pack all

    '''debug functions/control'''
    def increase_consumption(self):
        if self.consumption < max_c:                                                 # max consumption
            self.consumption += 1
            self.canvas.dchars(self.cons, 0, tk.END)
            self.canvas.insert(self.cons, 0, str(self.consumption))     
    def decrease_consumption(self):
        if self.consumption > min_c:                                                 # min consumption
            self.consumption += -1                                            # Adjust self.consumption with amount
            self.canvas.dchars(self.cons, 0, tk.END)                       # Delete all chars in self.cons
            self.canvas.insert(self.cons, 0, str(self.consumption))     # Insert new text in self.cons
    def increase_1power(self):
        if self.val_1power < max_p_1:
            self.val_1power += 10
            self.canvas.dchars(self.power1, 0, tk.END)
            self.canvas.insert(self.power1, 0, str(self.val_1power))
    def decrease_1power(self):
        if self.val_1power > min_p_1:
            self.val_1power  += -10
            self.canvas.dchars(self.power1 , 0, tk.END)
            self.canvas.insert(self.power1, 0, str(self.val_1power ))
    def increase_2power(self):
        if self.val_2power < max_p_2:
            self.val_2power += 10
            self.canvas.dchars(self.power2, 0, tk.END)
            self.canvas.insert(self.power2, 0, str(self.val_2power ))
    def decrease_2power(self):
        if self.val_2power > min_p_2:
            self.val_2power += -10
            self.canvas.dchars(self.power2, 0, tk.END)
            self.canvas.insert(self.power2, 0, str(self.val_2power))
 
    def update_percent(self, current_percent):
        self.canvas.dchars(self.edication_persent , 0, tk.END)
        self.canvas.insert(self.edication_persent , 0, str(current_percent))     
        

    def reset(self):    # return observation
        self.update()
        #time.sleep(0.5)
        self.canvas.dchars(self.power1, 0, tk.END)                       # Delete chars in self.power1,2
        self.canvas.dchars(self.power2, 0, tk.END)
        self.canvas.insert(self.power1, 0, str(self.val_1power))       # Insetr new text in self.power1,2
        self.canvas.insert(self.power2, 0, str(self.val_2power))

        #debug:
        #print("RESETED")
        #print(np.array([((self.val_1power+self.val_2power)-self.consumption)]))
       
        # return observation:
        #return np.array([self.val_1power/k_power, self.val_2power/k_power, ((self.val_1power+self.val_2power)-self.consumption)/k_freq])              # state (+ normalization)
        #return k_freq*np.array([(np.array(self.val_1power+self.val_2power) - np.array(self.consumption)) / (self.val_1power+self.val_2power+self.consumption)])
        return k_freq*np.array([(self.val_1power+self.val_2power) - (self.consumption)])

       
    def step(self, action):
        #s = (np.array([((self.val_1power+self.val_2power)-self.consumption)]))  # state (nparray variant)
        #s = np.array([self.val_1power/k_power, self.val_2power/k_power, ((self.val_1power+self.val_2power)-self.consumption)/k_freq])              # state (+ normalization)
        #s = k_freq*np.array([(np.array(self.val_1power+self.val_2power) - np.array(self.consumption)) / (self.val_1power+self.val_2power+self.consumption)])
        s = k_freq*np.array([(self.val_1power+self.val_2power) - (self.consumption)])

        #base_action = np.array([0, 0])   
        base_action = [0, 0]            # up/down 1/2 station

        '''action is the output neurons
         choose action: '''
        if action == 0:                       # up 1 station
            if self.val_1power<max_p_1:           # max power   
                base_action[0] += 1
            else:
                base_action[0] += 0
        elif action == 1:                    # down  1 station
            if self.val_1power>min_p_1:           # min power     
                base_action[0] += -1
            else:
                base_action[0] += 0
        elif action == 2:                    # up 2 station    
            if self.val_2power<max_p_2:              # max power    
                base_action[1] += 5
            else:
                base_action[1] += 0
        elif action == 3:                    # up 2 station
            if self.val_2power>min_p_2:              # min power    
                base_action[1] += -5
            else:
                base_action[1] += 0
        elif action == 4:                     # no action
            base_action[0] = 0    
            base_action[1] = 0
            
        '''CHANGE THE GENERATION VALUES'''
                                                            
        self.val_1power += base_action[0]
        self.canvas.dchars(self.power1, 0, tk.END)
        self.canvas.insert(self.power1, 0, str(self.val_1power)) 
        
        self.val_2power += base_action[1]
        self.canvas.dchars(self.power2, 0, tk.END)
        self.canvas.insert(self.power2, 0, str(self.val_2power))
        
       
        '''REFRESH FREQUENCY'''
        self.frequency= 50 + ((self.val_1power+self.val_2power) - (self.consumption)) / (self.val_1power+self.val_2power+self.consumption)*100   # static load characteristic
        self.canvas.dchars(self.freq, 0, tk.END)
        self.canvas.insert(self.freq, 0, str(self.frequency)) 
        #print('frequency = ', self.frequency)

        
        next_coords = (self.val_1power+self.val_2power)  
        reward_len = ((self.val_1power+self.val_2power) - (self.consumption))/(self.val_1power+self.val_2power +self.consumption)   

        # REWARD FUNCTION
        if next_coords == self.consumption:  # balace of consumption and generation  
            reward = 0.05
        else:
            reward = -0.01

        #s_ = k_freq*np.array([(np.array(self.val_1power+self.val_2power) - np.array(self.consumption)) / (self.val_1power+self.val_2power+self.consumption)])
        #s_ = np.array([self.val_1power/k_power, self.val_2power/k_power,((self.val_1power+self.val_2power)-self.consumption)/k_freq])              # state (+ normalization)
        s_ = k_freq*np.array([(self.val_1power+self.val_2power) - (self.consumption)])
        #print("s_ = ", s_)

        
        return s_, reward

    def render(self):
        #time.sleep(0.1)
        self.update()

        

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(self):
        self.n_actions = len(act)
        self.n_features = n_f
        self.n_neurons_hidden_layer = n_n
        self.lr = l_r
        self.gamma = r_w
        self.replace_target_iter = r_t_i
        self.memory_size = m_s
        self.batch_size = b_s
        self.epsilon_increment = e_g_enc
        #self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon = epsil
        # total learning step
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))   #  2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        # consist of [target_net, evaluate_net]
        self._build_net()    # build net arfter init

        with tf.variable_scope('soft_replacement'):  # 
            self.target_with_evolve_replace = [tf.assign(t, e) for t, e in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                                 scope='target_net'),
                                                                                                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                                                 scope='eval_net'))]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        if output_graph:
            tf.summary.FileWriter("C:/tensorboard/DQN/", self.sess.graph)
        self.cost_his = []


        

    def decrease_greedy(self):
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < 1.0 else 1.0
        self.learn_step_counter += 1
    def decrease_learning_rate(self):
        self.lr=self.lr*0.99
    def _build_net(self):

        ''' __________________ inputs _________________________'''

        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  #  State / count of input neurons
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        weights = tf.random_normal_initializer(0., 0.3)

        if bias_on == True:
            bias = tf.constant_initializer(0.1)
            ''' ____________________________ evaluate_net ____________________________________'''
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, units=self.n_neurons_hidden_layer, activation=tf.nn.relu, kernel_initializer=weights, bias_initializer=bias, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=weights, bias_initializer=bias, name='q')
            ''' ______________________________ target_net ______________________________________'''
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, units=self.n_neurons_hidden_layer, activation=tf.nn.relu, kernel_initializer=weights, bias_initializer=bias, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=weights, bias_initializer=bias, name='t2')
        else:
            ''' ____________________________ evaluate_net ____________________________________'''
            with tf.variable_scope('eval_net'):
                e1 = tf.layers.dense(self.s, units=self.n_neurons_hidden_layer, activation=tf.nn.relu, kernel_initializer=weights, name='e1')
                self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=weights, name='q')
            ''' ______________________________ target_net ______________________________________'''
            with tf.variable_scope('target_net'):
                t1 = tf.layers.dense(self.s_, units=self.n_neurons_hidden_layer, activation=tf.nn.relu, kernel_initializer=weights, name='t1')
                self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=weights, name='t2')

        # Bellman's equation: Q(s, a) = r(s, a) + yQ(s', a)
        # s - current state (q_eval), s' - next state (q_next)
        # Q(s, a) calculating by target net:        
        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, ) reduce to vector with shape = axis
            #q_target = self.r + self.gamma * np.amax(self.q_next, axis = 1) np.max or tf.reduce_max
            self.q_target = tf.stop_gradient(q_target) # no calculating gradient in this net (wheights improwed by "self.target_with_evolve_replace")
            
        # q_eval give us the actions of neural net in current state (s):
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )


        with tf.variable_scope('loss'):
            #self.loss =S tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))  # origina
            self.loss = tf.reduce_mean((self.q_target - self.q_eval_wrt_a)**2)
        with tf.variable_scope('train'):                                           
             self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)                    
             #self._train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
             #self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
             #self._train_op = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)
             #self._train_op =tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
             #self._train_op =tf.train.FtrlOptimizer(self.lr).minimize(self.loss)
             #self._train_op =tf.train.MomentumOptimizer(self.lr).minimize(self.loss)
             #self._train_op =tf.train.ProximalAdagradOptimizer(self.lr).minimize(self.loss)
             #self._train_op =tf.train.ProximalGradientDescentOptimizer(self.lr).minimize(self.loss)
            
            #print(self.loss)

    def store_transition(self, s, a, r, s_):  # replay memory
        if not hasattr(self, 'memory_counter'):    #
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        #print('stored')

        
    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # to have batch dimension when feed into tf placeholder
        if np.random.uniform() < self.epsilon:
            # feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(self.n_actions)
        #print('e-greedy = ', self.epsilon)
        #print('action',action)    
        return action
 
    '''
      
     # old 

    def _replace_target_params(self):
        targNET_param = tf.get_collection('taregt_net_params')
        evalNET_param = tf.get_collection('taregt_net_params')
        self.sess.run([tf.assign(t,e) for t, e in zip{targNET_param, evalNET_param}])
    '''

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:      # replace target net after # iterations
            self.sess.run(self.target_with_evolve_replace)     
            #print(self.learn_step_counter, '\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run( [self._train_op, self.loss], feed_dict={self.s: batch_memory[:, :self.n_features], self.a: batch_memory[:, self.n_features], self.r: batch_memory[:, self.n_features + 1], self.s_: batch_memory[:, -self.n_features:],})

        self.cost_his.append(cost)
        self.learn_step_counter += 1   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def run_env():
    step = 0

    for episode in range(1):   # 
        # initial observation
        observation = env.reset()

        while step <= 5000:
        #while True:
            # refresh
            env.render()

            if random_consumption_on == True: 
                rand = random.random()
                if (rand<0.1):
                    env.increase_consumption()
                elif (rand>0.9):
                    env.decrease_consumption()
            
            # RL choose action based on observation

            action = RL.choose_action(observation)
            # RL take action and get next observation/reward
            observation_, reward = env.step(action)
            
            RL.store_transition(observation, action, reward, observation_)

            if (step > 100 and step % 5 == 0):
                RL.learn()
               
            # swap observation
            observation = observation_
            
            #if (step > 1000 and step % 100 == 0):
             #   RL.decrease_learning_rate()
                
            if (step > start_increase_e_greedy and step % 100 == 0):
                RL.decrease_greedy()
            
            file1 = open(r"./DQN_step.txt", "a")
            file1.write(str(step) + ';')
            file1.close
            file2 = open(r"./DQN_freq.txt", "a")
            file2.write(str(env.frequency) + ';')
            file2.close

            
            step += 1
            #print('step: ', step)
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        a = open('./DQN_step.txt', 'r')
        b=a.read().split(';')
        b.pop()
        x=[float(i) for i in b]
        c = open('./DQN_freq.txt', 'r')
        d=c.read().split(';')
        d.pop()
        y=[float(i) for i in d]
        fig, ax = plt.subplots()
        ax.plot(x,y, color = "red", label="DQN")
        ax.set_xlabel("step")
        ax.set_ylabel("frequency")
        ax.legend()
        fig.savefig('./DQN.png')


if __name__ == "__main__":
    # maze game
    env = Sys()
    RL = DeepQNetwork()
    env.after(100, run_env())
    env.mainloop()

