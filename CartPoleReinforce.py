import gym
import torch
import torch.nn as nn
import numpy as np
import math

class Network(nn.Module):
    
    def __init__(self):
        super(Network, self).__init__()
        self.num_inputs = 4
        self.num_outputs = 2
        self.output1 = nn.Linear(self.num_inputs, self.num_outputs)
        self.output1.bias.data = torch.tensor([float(0), float(0)])
        self.real_output = nn.Softmax(dim=-1)
        self.alpha = 0.5

    def getParameters(self):
        return self.output1.weight

    def updateLearningParameter(self, num_episodes, avg):
        self.alpha = max(0.5 * pow(0.5, int(num_episodes/avg)), 0.005)

    def getLearningParameter(self):
        return self.alpha

    def changeParameters(self, log_grad, tot_return):
        for i in range(self.num_outputs):
            for j in range(self.num_inputs):
                log_grad[i][j] = self.output1.weight[i][j] + log_grad[i][j]*self.alpha*tot_return
        self.output1.weight.data = log_grad.data
        self.output1.weight.grad = None
        
    def simulate_forward(self, pos, vel, angle, tip_vel):
        pairs = torch.tensor([pos, vel, angle, tip_vel])
        output1 = self.output1(pairs)
        real = self.real_output(output1)
        return real, output1

    def forward(self, pos, vel, angle, tip_vel):
        pairs = torch.tensor([pos, vel, angle, tip_vel], requires_grad=True)
        self.output1.weight.retain_grad()
        output1 = self.output1(pairs)
        real = self.real_output(output1)
        return real, output1

    def log_gradients(self, softmax_layer, output_layer, action):
        log_output = torch.log(softmax_layer)
        log_output[action].backward()
        return self.output1.weight.grad
        

env_name = "CartPole-v0"
env = gym.make(env_name)
max_action= 2
num_episodes = 10000
avg = 50

net = Network()
tot_reward = 0
amount_solves = 0

for episode in range(num_episodes):
    state = env.reset()

    t = 0
    reward_list = []
    state_list = [state]
    action_list = []
    reward_sum = 0
    while(1):
        softmax_l, output_l = net.simulate_forward(np.float32(state[0]), np.float32(state[1]), np.float32(state[2]), np.float32(state[3]))

        curr_sum = float(0)
        action = -1
        random_value = np.random.random_sample()

        if(random_value <= softmax_l[0]):
            action = 0
        elif(random_value > softmax_l[0]):
            action = 1
        
        #print(random_value, softmax_l, action)
        if(action == -1):
            print(random_value, softmax_l, state)
        state, reward, done, _ = env.step(action)
        #print(state, reward, done)
        
        reward_list.append(reward)
        state_list.append(state)
        action_list.append(action)
        reward_sum += reward
        
        if(done):
            break

    if(reward_sum >= 200):
        amount_solves += 1
    tot_reward += reward_sum
    
    if((episode+1) % avg == 0):
        print("Episode: ", (episode+1)/avg, " Reward: ", tot_reward/avg)
        #print("Amount solved: ", amount_solves)
        print("Weights: ", net.getParameters())
        print("Alpha: ", net.getLearningParameter())
        print()
        tot_reward = 0
        amount_solves = 0
    for i in range(len(action_list)):
        s, o = net.forward(np.float32(state_list[i][0]), np.float32(state_list[i][1]), np.float32(state_list[i][2]), np.float32(state_list[i][3]))
        log_grad = net.log_gradients(s, o, action_list[i])
        net.changeParameters(log_grad, reward_sum)
        reward_sum -= reward_list[i]
    net.updateLearningParameter(episode, 50)
    

