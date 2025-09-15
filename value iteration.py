import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
class State:
    def __init__(self,x,y,list=[],):
        self.x=x
        self.y=y

class robot:
    def __init__(self,x_start,y_start,x_end,y_end):
        self.x_start=x_start
        self.y_start=y_start
        self.x_end=x_end
        self.y_end=y_end
        self.inter=[]
        self.pathx=[]
        self.pathy=[]
        self.stop=False
        self.terminal=False
        self.battery=100
    def move(self,action,step):
        
        if self.reach_for_multi():
            self.battery=100
            self.battery=self.battery-0.1*step

        else:
           
           
            self.battery=self.battery-0.1*step
            if action==0:
                self.y_position=self.y_position+step
                self.x_position=self.x_position
            elif action==1:
                self.x_position=self.x_position+step
                self.y_position=self.y_position+step
            elif action==2:
                self.y_position=self.y_position
                self.x_position=self.x_position+step 
            elif action==3:
                self.x_position=self.x_position+step
                self.y_position=self.y_position-step
            elif action==4:
                self.x_position=self.x_position
                self.y_position=self.y_position-step
            elif action==5:
                self.x_position=self.x_position-step
                self.y_position=self.y_position-step
            elif action==6:
                self.x_position=self.x_position-step
                self.y_position=self.y_position
            elif action==7:
                self.x_position=self.x_position-step
                self.y_position=self.y_position+step
        return  self.x_position,self.y_position    
    def get_policy(self,policy):
            self.policy=policy    
    def getnext_move(self,action,step):
            x=self.x_position
            y=self.y_position
            if action==0:
                y=y+step
                x=x
            elif action==1:
                x=x+step
                y=y+step
            elif action==2:
                y=y
                x=x+step 
            elif action==3:
                x=x+step
                y=y-step
            elif action==4:
                x=x
                y=y-step
            elif action==5:
                x=x-step
                y=y+step
            elif action==6:
                x=x-step
                y=y
            elif action==7:
                x=x-step
                y=y+step
            return  x,y 
    def get(self,step):
        return int(self.x_position-step/2), int(self.y_position-step/2)

    def modify(self,step):
        self.x_start+=step/2
        self.y_start+=step/2
        self.x_end+=step/2
        self.y_end+=step/2
        self.x_position=self.x_start
        self.y_position=self.y_start
    def reach_for_multi(self):
        if self.x_end==self.x_position and self.y_end==self.y_position:
            return True
        else:
            return False     
    def reach_goal(self,state):
        if self.x_end==state.x and self.y_end==state.y:
            return True
        else:
            return False
class Environment:
    def __init__(self,width,height,list,step):
        self.step=step

        self.grid_dimenson=[]
        self.grid_dimenson.append(width+self.step)
        self.grid_dimenson.append(height+self.step)
        
        self.obstacles_list=list
        for i in range(0,len(self.obstacles_list),4):


            self.obstacles_list[i]+=self.step/2
            self.obstacles_list[i+1]+=self.step/2
    def get_nexstate(self,state,action):
        next_state=State(0,0)
        

        if action==0:
            next_state.y=state.y+self.step
            next_state.x=state.x
        elif action==1:
            next_state.x=state.x+self.step
            next_state.y=state.y+self.step
        elif action==2:
            next_state.y=state.y
            next_state.x=state.x+self.step 
        elif action==3:
            next_state.x=state.x+self.step
            next_state.y=state.y-self.step
        elif action==4:
            next_state.x=state.x
            next_state.y=state.y-self.step
        elif action==5:
            next_state.x=state.x-self.step
            next_state.y=state.y-self.step
        elif action==6:
            next_state.x=state.x-self.step
            next_state.y=state.y
        elif action==7:
            next_state.x=state.x-self.step
            next_state.y=state.y+self.step
        

        return next_state 
    def valid_way(self,state,agent):
        test_state=State(0,0)
        status=True
        for t in np.arange(0, 1, 0.05):
            test_state.x=(1-t)*state.x+t*agent.x_end
            test_state.y=(1-t)*state.y+t*agent.y_end
            if not self.valid_state(test_state):
                status=False
                break
        return status    



    def compute_reward(self,state,agent,beta): 
        reward=-1
        distance_to_goal = np.sqrt((state.x - agent.x_end)**2 + (state.y - agent.y_end)**2) 
        reward=reward-beta*distance_to_goal

        if agent.reach_goal(state):
            reward=reward+1000
                

            
        if  not self.valid_state(state):
            reward=reward-1000
               

           
        return reward  
    def valid_state(self,state):
        valid=True
        for i in range(0,len(self.obstacles_list),4):
            xob1=self.obstacles_list[i]-self.obstacles_list[i+2]/2
            xob2=self.obstacles_list[i]+self.obstacles_list[i+2]/2
            yob1=self.obstacles_list[i+1]-self.obstacles_list[i+3]/2
            yob2=self.obstacles_list[i+1]+self.obstacles_list[i+3]/2
            if state.x>=xob1 and state.x<=xob2 and state.y>=yob1 and state.y<=yob2 :
                valid=False
                break
        if valid==True:
            if  self.is_Out(state):
                valid=False 
        return valid   
    def is_Out(self,state):
        out=True
        if state.x>self.grid_dimenson[0] or state.x<0 or state.y>self.grid_dimenson[1] or state.y<0:
            out=True
        else:
            out=False
        return out    
    def get_state(self,state):
        return state.x-self.step/2,state.y-self.step/2
    
        
              
       
class RL:
    def __init__(self,agent,environment):

        self.agent=agent
        self.environment=environment
        self.state_space=[[0 for _ in range(int(self.environment.grid_dimenson[1]))] for _ in range(int(self.environment.grid_dimenson[0]))]
        self.policy=np.zeros((int(self.environment.grid_dimenson[0]) ,int(self.environment.grid_dimenson[1])))
        self.grid_states=np.zeros((int(self.environment.grid_dimenson[0]) ,int(self.environment.grid_dimenson[1]))) 
        for i in range(0,int(self.environment.grid_dimenson[0])):
            for j in range(0,int(self.environment.grid_dimenson[1])):
                self.state_space[i][j]=State(self.environment.step*i+self.environment.step/2,self.environment.step*j+self.environment.step/2) 
           
        self.action_space=[0,1,2,3,4,5,6,7]
        self.agent.modify(self.environment.step)
    def get_state_value(self,state):
        ret=0
        if not self.environment.valid_state(state):
            
            ret=-sys.float_info.max
        else:
            
            a,b=self.environment.get_state(state)
            ret=self.grid_states[int(a)][int(b)]
        return ret    
    def value_iteration(self,gamma,beta):
        
        
        threshold = 1e-7 
        verror = float('inf')  
        count=1
        while verror > threshold:
            new_values = np.copy(self.grid_states) 
            verror = 0 
            
            for i in range(self.environment.grid_dimenson[0]):
                for j in range(self.environment.grid_dimenson[1]):
                    
                    if self.agent.reach_goal(self.state_space[i][j]):  
                        continue  
                    
                    max_value = float('-inf')
                    best_action = None

                    for action in self.action_space:
                        next_state = self.environment.get_nexstate(self.state_space[i][j], action)
                        reward = self.environment.compute_reward(next_state, self.agent,beta)
                        
                        value = (1/8) * (reward + gamma * self.get_state_value(next_state))
                        
                        if value > max_value:
                            max_value = value
                            best_action = action
                    
                    
                    new_values[i][j] = max_value
                    self.policy[i][j] = best_action
                    
                    
                    verror = max(verror, abs(new_values[i][j] - self.grid_states[i][j]))

            count=count+1
            self.grid_states = new_values
        self.agent.get_policy(self.policy) 
           
        return self.policy, self.grid_states,count 
    def cumulative(self,agent,beta):
        reach=False
        sum=0
        state=State(agent.x_start,agent.y_start)
        previous=state
        while reach==False:
            
            a,b=self.environment.get_state(state)
            action=agent.policy[int(a)][int(b)]
            next_state=self.environment.get_nexstate(state,action)
            reward=self.environment.compute_reward(next_state,agent,beta)
            sum+=reward
            if self.agent.reach_goal(next_state):
                reach=True
            if next_state.x==previous.x and next_state.y==previous.y:
                sum=-1000
                break    
            previous=state
               
            state=next_state

            
        return sum        
    def plot_policy_grid(self):
        fig, ax = plt.subplots(figsize=( len(self.grid_states),len(self.grid_states[0]))) 
        ax.set_xlim(-0.5, len(self.grid_states) - 0.5)
        ax.set_ylim(-0.5, len(self.grid_states[0]) - 0.5)

       
        action_arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']
        action_vectors = [
            (0, 0.4),   
            (0.4, 0.4), 
            (0.4, 0),   
            (0.4, -0.4),
            (0, -0.4),  
            (-0.4, -0.4),
            (-0.4, 0),  
            (-0.4, 0.4) 
        ]

        
        for i in range(len(self.grid_states)):
            for j in range(len(self.grid_states[i])):
                state = self.state_space[i][j]

                
                if not self.environment.valid_state(state):
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color='gray', alpha=0.8))
                elif state.x == self.agent.x_end and state.y == self.agent.y_end:
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color='black', alpha=0.5)) 
                    ax.text(i, j, 'G', fontsize=16, ha='center', va='center', color='black')    
                else:
                    action = int(self.policy[i][j])  
                    dx, dy = action_vectors[action]  
                    ax.arrow(i, j, dx, dy, head_width=0.1, head_length=0.1, fc='black', ec='black')    
                if state.x == self.agent.x_start and state.y == self.agent.y_start:
                    ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color='white', alpha=0.5)) 
                    ax.text(i, j, 'S', fontsize=16, ha='center', va='center', color='black')

              
                

                

      
        ax.set_xticks(np.arange(-0.5, len(self.grid_states), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(self.grid_states[0]), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Policy Grid")

        plt.show()
def run_multi_robots(agents,environment,alpha,beta):
    for i in range(len(agents)):
        
        Rll=RL(agents[i],environment) 
        Rll.value_iteration(alpha,beta)
        Rll.plot_policy_grid()
    listx=[]
    listy=[]

    for i in range(12):
        for i in range(len(agents)):
            if not agents[i].reach_for_multi():
                a,b=agents[i].get(environment.step)
                x,y=agents[i].getnext_move(agents[i].policy[a][b],environment.step)
                listx.append(x)
                listy.append(y)

            else :
                listx.append(agents[i].x_position)
                listy.append(agents[i].y_position)
            
        for i in range(len(agents)):
            for j in range(i+1,len(agents),1):
                if listx[i]==listx[j]  and   listy[i]==listy[j] :
                    agents[i].inter.append(j)
        for i in range(len(agents)):
                agents[i].pathx.append(agents[i].x_position)
                agents[i].pathy.append(agents[i].y_position)            
        for i in range(len(agents)):
            minb=agents[i].battery
            ind=i

            
            for j in range(len(agents[i].inter)):
                if agents[agents[i].inter[j]].battery<minb:
                    minb=agents[j].battery
                    ind=j
                else :
                    agents[agents[i].inter[j]].stop=True

            
            if agents[i].stop!=True and not agents[ind].reach_for_multi():
                    a,b=agents[ind].get(environment.step)
                    agents[ind].move(agents[ind].policy[a][b],environment.step)
            agents[i].inter=[]
        for i in range(len(agents)):
            agents[i].stop=False
        listx=[]
        listy=[]  
def animate_paths(agents, environment, max_steps=12):
    
    fig, ax = plt.subplots(figsize=(environment.grid_dimenson[0], environment.grid_dimenson[1]))

    
    for i in range(0, len(environment.obstacles_list), 4):
        xob, yob = environment.obstacles_list[i], environment.obstacles_list[i + 1]
        wob, hob = environment.obstacles_list[i + 2], environment.obstacles_list[i + 3]
        ax.add_patch(plt.Rectangle((xob-wob/2, yob-hob/2), environment.obstacles_list[i + 2], environment.obstacles_list[i + 3], color='gray', alpha=0.8))

    
    for agent in agents:
        ax.text(agent.x_start, agent.y_start, 'S', fontsize=14, ha='center', va='center', color='green')
        ax.text(agent.x_end, agent.y_end, 'G', fontsize=14, ha='center', va='center', color='red')

    ax.set_xlim(0, environment.grid_dimenson[0])
    ax.set_ylim(0, environment.grid_dimenson[1])
    ax.set_xticks(np.arange(0, environment.grid_dimenson[0] + environment.step, environment.step))
    ax.set_yticks(np.arange(0, environment.grid_dimenson[1] + environment.step, environment.step))
    ax.grid(which='both', linestyle='--', linewidth=0.5)

   
    robot_markers = [ax.plot([], [], 'o', markersize=10)[0] for _ in agents]

    def init():
     
        for marker in robot_markers:
            marker.set_data([], [])
        return robot_markers

    def update(frame):
       
        for i, agent in enumerate(agents):
            if frame < len(agent.pathx):  # Ensure within range
                robot_markers[i].set_data(agent.pathx[frame], agent.pathy[frame])
        return robot_markers

    ani = animation.FuncAnimation(fig, update, frames=max_steps, init_func=init, blit=True, interval=500, repeat=False)
    plt.show() 
def plot_agents_paths(agents, environment):
  
    fig, ax = plt.subplots(figsize=(environment.grid_dimenson[0], environment.grid_dimenson[1]))

   
    for i in range(0, len(environment.obstacles_list), 4):
        xob, yob = environment.obstacles_list[i], environment.obstacles_list[i + 1]
        wob, hob = environment.obstacles_list[i + 2], environment.obstacles_list[i + 3]
        ax.add_patch(plt.Rectangle((xob - wob / 2, yob - hob / 2), wob, hob, color='gray', alpha=0.8))

   
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    
    for i, agent in enumerate(agents):
        color = colors[i % len(colors)]  

        
        ax.plot(agent.pathx, agent.pathy, linestyle='-', marker='o', color=color, markersize=5, linewidth=2, label=f"Agent {i+1}")

       
        ax.text(agent.x_start, agent.y_start, 'S', fontsize=14, ha='center', va='center', color='black', fontweight='bold')
        ax.text(agent.x_end, agent.y_end, 'G', fontsize=14, ha='center', va='center', color='black', fontweight='bold')

    
    ax.set_xlim(0, environment.grid_dimenson[0])
    ax.set_ylim(0, environment.grid_dimenson[1])
    ax.set_xticks(np.arange(0, environment.grid_dimenson[0] + environment.step, environment.step))
    ax.set_yticks(np.arange(0, environment.grid_dimenson[1] + environment.step, environment.step))
    ax.grid(which='both', linestyle='--', linewidth=0.5)

   
    ax.set_title("Agents' Paths on Grid")
    ax.legend()
    
    
    plt.show()
                        
width=10
height=10
step=1
list=[]


list.append(2)
list.append(4)
list.append(1)
list.append(9)

list.append(4)
list.append(6)
list.append(1)
list.append(9)
list.append(6)
list.append(4)
list.append(1)
list.append(9)
list.append(8)
list.append(6)
list.append(1)
list.append(9)
list.append(9)
list.append(4)
list.append(1)
list.append(1)
list.append(10)
list.append(5)
list.append(1)
list.append(1)
list.append(9)
list.append(6)
list.append(1)
list.append(1)
list.append(10)
list.append(7)
list.append(1)
list.append(1)

x_start=1
y_start=1
x_end=9
y_end=9
robot1=robot(x_start,y_start,x_end,y_end)
robot2=robot(9,1,1,9)
robot3=robot(1,1,9,9)

agents=[robot1]
environment=Environment(width,height,list,step)   
run_multi_robots(agents,environment,0.9,0.1)


plot_agents_paths(agents, environment)


        
        
        
      

 
    

       
                