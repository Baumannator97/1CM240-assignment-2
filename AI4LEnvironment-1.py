"""
Spyder Editor

This is a temporary script file.
"""
import gym
from gym import spaces
import numpy as np
import geopy.distance

#import matplotlib.pyplot as plt

#from tabulate import tabulate
#from vrplib import read_solution

from pyvrp import Model#, read
#from pyvrp.plotting import (
#    plot_coordinates,
#    plot_instance,
#    plot_result,
#    plot_route_schedule,
#)
from pyvrp.stop import MaxIterations, MaxRuntime



#This is the openAI gym implementation of our assignment.
class AI4LEnvironment(gym.Env):
    """Dynamic Inventory Routing Environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super().__init__()
        self.nStores = 17

        self.data = dict()
        
        ## latitute coordinates; First element is depot
        self.lat = np.array([52.4572735973313,
                             52.5626752866663,
                             52.5524998075759,
                             52.5485533897899,
                             52.5491337603554,
                             52.533031250357,
                             52.5326620602486,
                             52.5257945584331,
                             52.5360673338073,
                             52.513899889604,
                             52.5006751919937,
                             52.4805171338363,
                             52.4965099365104,
                             52.4921203344399,
                             52.4575252353351,
                             52.4873876246243,
                             52.4976166075393,
                             52.4861757546093
                             ])
        
        ## Longitude coordinates, first element is depot
        self.lon = np.array([13.3878670887734,
                             13.364101495398,
                             13.3610115906129,
                             13.4127270662978,
                             13.4547634845713,
                             13.387585268824,
                             13.398873880729,
                             13.4156247242011,
                             13.435584982625,
                             13.4691377305421,
                             13.4760565571936,
                             13.4389650463,
                             13.4224855541126,
                             13.4226572154896,
                             13.3920890219171,
                             13.3764057793401,
                             13.3456685769195,
                             13.3199641186209
                             ])
        
        
        self.data['distance_matrix'] = np.zeros(shape = [self.nStores + 1, self.nStores + 1])
        
        #variable transport cost per unit distance
        self.transportCost = 5;
        
        #fixed transport cost per used route
        self.fixedTransportCost = 100;
        
        
        #the distance matrix is in the end the cost matrix. we take the distance
        #between coordinates and multiply it with the variable transportCost
        for i in range(0, self.nStores + 1):
            for j in range(0, self.nStores + 1):
                coords_1 = (self.lat[i], self.lon[i])
                coords_2 = (self.lat[j], self.lon[j])
                
                self.data['distance_matrix'][i][j] = geopy.distance.geodesic(coords_1, coords_2).km * self.transportCost
                                           
                            
        # the vehicle capacity, the number of vehicles.
        self.data['vehicle_capacity'] = 50
        self.data['num_vehicles'] = 17

        #Information of the stores (holding cost, lost-sales cost, capacity)
        self.c_holding = 1 
        self.c_lost    = 10
        self.capacity  = 1000
        
        # The maximum to be shipped to a store
        self.maxOrderQuantity = 1000;
                
        # the current amount of inventory in each store
        self.inventories = np.zeros(self.nStores + 1)
               
        
        
        
        # THIS IS FAKE DEMAND YOU HAVE TO IMPROVE UPON THIS !!!!!!
        
        self.demandMean = np.array([ 0, #fake depot entry to let indices match
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10, 
                                    10,
                                    10, 
                                    10,
                                    10, 
                                    10])
        
        np.random.seed(1331)
                
        # create some fixed order up to levels. 
        # THIS IS JUST FOR ILLUSTRATION PURPOSES
        self.orderUpTo = np.ceil(self.demandMean + 2)


        #For bookkeeping purposes
        self.demands = np.zeros(self.nStores + 1)
        self.action  = np.zeros(self.nStores + 1)
        self.cost    = 0
        self.avgCost = 0;
        
        
        
        #OPEN AI GYM elements that need to be set
        #this should indicate between which values the rewards could fluctuate
        # (Your teacher has no real clue what happens with it)
        self.reward_range = (self.nStores * -1 * self.capacity * self.c_lost, 3 * self.capacity * self.c_holding)
        
        # we need to define the shape of an action
        # for this example, we set it equal to a simple multibinairy action
        # space. (series of zeros and ones for ordering or not)
        # It is quite crucial to understand the spaces objects. Please google!
        
        # Also note that this action is ignored as we use a base stock
        # a first step towards implementation could be to ignore visiting 
        # a store.
        
        
        
        # how many stores we will replenish to base stock?
        self.action_space = spaces.Discrete(self.nStores + 1)
        
        
        #observation space is simply the inventory levels at each store at the
        #start of the day
        self.observation_space = spaces.Box(low = 0, 
                                            high = self.capacity, 
                                            shape = (self.nStores + 1,),
                                            dtype = np.int)        
        
    
    def calcDirectReward(self, action):
        
               
        self.data['demands'] = self.orderUpTo - self.inventories            
        
        
        m = Model()
        
        int_lat = [int(number * 100) for number in self.lat]
        int_lon = [int(number * 100) for number in self.lon]

        
        depot = m.add_depot(x=int_lat[0], y=int_lon[0])
        clients = [
            m.add_client(x= int_lat[idx], y=int_lon[idx], demand=self.data['demands'][idx])
            for idx in range(1, len(int_lat))
        ]
                
        locations = [depot] + clients
        
        m.add_vehicle_type(self.data['vehicle_capacity'], self.nStores, self.fixedTransportCost)
        
        for idx in range(0, len(self.lat)): 
            for jdx in range(0, len(self.lat)):
                distance = self.data['distance_matrix'][idx][jdx]   # Manhattan
                m.add_edge(locations[idx], locations[jdx], distance=distance)
     
        
        
     
        res = m.solve(stop = MaxRuntime(0.0001))

        print(res)
     
        return -1 * res.cost()


        
    def step(self, action):
        # Execute one time step within the environment
        
        
        reward = self.calcDirectReward(action)

        self._take_action(action)
        self.current_step += 1
        
        # generate random demand
        demands = np.zeros(self.nStores)
        
        
        for i in range(0, self.nStores):
            demands[i] = self.demandMean[i]
            self.inventories[i] -= demands[i]
            reward -= max(0, self.inventories[i]) * self.c_holding + -1 * min(0, self.inventories[i]) * self.c_lost
            
            self.inventories[i] = max(0, self.inventories[i])
        
        
        
        self.cost += reward
        self.avgCost = self.cost / self.current_step
        
        
        done = self.current_step >= 2000
        
        obs = self._next_observation() 

        return obs, reward, done, {}
    
        
    def _take_action(self, action):
        
        #in this example it is rather simple; the inventory is shipped
        self.inventories = self.orderUpTo.copy()
        
        
        
          
    def reset(self):
        # Reset the state of the environment to an initial state
        self.inventories = np.zeros(self.nStores + 1)
                
        self.current_step = 0
        
        
        self.cost = 0
        self.avgCost = 0;
        
        return self._next_observation()
    
    
    def _next_observation(self):
        return self.inventories
        
    
    def render(self, mode='human', close=False):
        print("No rendering implemented")




def example(episodes=10):
    
    env = AI4LEnvironment()
    total_rewards = []

    for episode in range(episodes):
        observation = env.reset()
        observation, reward, done, info = env.step(0)
        total_rewards.append(reward)
      
        print(f'Episode {episode + 1}, Total Reward: {reward}')

    env.close()
    return total_rewards

# Train the agent
total_rewards = example()

# Simple analysis
print(f'Average reward: {sum(total_rewards) / len(total_rewards)}')






