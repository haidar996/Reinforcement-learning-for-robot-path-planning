import numpy as np
import math
import sys
import random
import matplotlib.pyplot as plt
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
    def modify(self,step):
        self.x_start=x_start+step/2
        self.y_start=y_start+step/2
        self.x_end=x_end+step/2
        self.y_end=y_end+step/2
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



    def compute_reward(self,state,agent): 
        reward=-1
        distance_to_goal = np.sqrt((state.x - agent.x_end)**2 + (state.y - agent.y_end)**2) 
        reward=reward-0.1*distance_to_goal

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
        return int(state.x-self.step/2),int(state.y-self.step/2)
    
        
              
       
class RL:
    def __init__(self,agent,environment):

        self.agent=agent
        self.experience=[]
        self.environment=environment
        self.state_space=[[0 for _ in range(int(self.environment.grid_dimenson[1]))] for _ in range(int(self.environment.grid_dimenson[0]))]
        self.policy=np.zeros((int(self.environment.grid_dimenson[0]) ,int(self.environment.grid_dimenson[1])))
        self.grid_states=np.zeros((int(self.environment.grid_dimenson[0]) ,int(self.environment.grid_dimenson[1]))) 
        self.last_visited=[[sys.float_info.max for _ in range(int(self.environment.grid_dimenson[1]))] for _ in range(int(self.environment.grid_dimenson[0]))]
        for i in range(0,int(self.environment.grid_dimenson[0])):
            for j in range(0,int(self.environment.grid_dimenson[1])):
                self.state_space[i][j]=State(self.environment.step*i+self.environment.step/2,self.environment.step*j+self.environment.step/2) 
           
        self.action_space=[0,1,2,3,4,5,6,7]
        self.agent.modify(self.environment.step)
        
        self.model=[[[[0 for _ in range(2)]for _ in range(8)] for _ in range(self.environment.grid_dimenson[0])] for _ in range(self.environment.grid_dimenson[1])]
        self.actionex= [[[] for _ in range(self.environment.grid_dimenson[1])] for _ in range(self.environment.grid_dimenson[0])]
        self.Qsa=np.zeros((int(self.environment.grid_dimenson[0]) ,int(self.environment.grid_dimenson[1]),len(self.action_space)))
    def get_state_value(self,state):
        ret=0
        if not self.environment.valid_state(state):
            
            ret=-sys.float_info.max
        else:
            
            a,b=self.environment.get_state(state)
            ret=self.grid_states[int(a)][int(b)]
        return ret
    def get_action_values(self,state):
        ret=0
        if not self.environment.valid_state(state):
            
            ret=-sys.float_info.max
        else:
            
            a,b=self.environment.get_state(state)
            ret=np.amax(self.Qsa[a][b])
        return ret
    def get_Action(self,state,epsilon):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(self.action_space)
        else:
            a,b=self.environment.get_state(state)
            return np.argmax(self.Qsa[a][b])     
    def dyna_qplus(self,num_episodes,gamma,epsilon,alpha,planning,kappa):
        time=0
        for i in range(num_episodes):
            reach=False
            r,c=self.environment.get_state(State(self.agent.x_start,self.agent.y_start))
            state=self.state_space[r][c]
            print(i)  
            while reach==False:
                action=self.get_Action(state,epsilon)
                next_state=self.environment.get_nexstate(state,action)       
                reward=self.environment.compute_reward(next_state,self.agent)
                if self.agent.reach_goal(next_state):
                    reach=True
                r1,c1=self.environment.get_state(state)
                r2,c2=self.environment.get_state(next_state)    
                target=self.get_action_values(next_state)
                self.Qsa[r1][c1][action]+=alpha*(reward+gamma*target-self.Qsa[r1][c1][action])
                self.experience.append(state)
                self.model[r1][c1][action][0]=reward
                self.model[r1][c1][action][1]=State(next_state.x, next_state.y)
                
                
                self.actionex[r1][c1].append(action)
                self.last_visited[r1][c1]=time
                if  self.environment.valid_state(next_state):
                   state=next_state
                time+=10  
                if i ==int(num_episodes/10):

                    
                    for i in range(0,self.environment.grid_dimenson[0]):
                        for j in range(0,self.environment.grid_dimenson[1]):
                             self.policy[i][j]=np.argmax(self.Qsa[i][j])

                    self.plot_policy_grid()
                    self.environment.obstacles_list[6]=1
                    self.environment.obstacles_list[7]=5
                liss=self.last_visited    
                for j in range(planning):
                        if j>planning/3:
                            
                            minn=sys.float_info.max
                            for r in range(0,self.environment.grid_dimenson[0]):
                                for c in range(0,self.environment.grid_dimenson[1]):
                                    if minn>=liss[r][c]:
                                        minn=liss[r][c]
                                        a=r
                                        b=c
                            mostate=State(a+self.environment.step/2,b+self.environment.step/2)
                            bi=minn
                        else:
                            mostate=random.choice(self.experience)
                            u3,i3=self.environment.get_state(mostate)
                            bi=self.last_visited[u3][i3]
                        u1,i1=self.environment.get_state(mostate)
                        moaction=random.choice(self.actionex[u1][i1])
                    
                        moreward=self.model[u1][i1][moaction][0]
                        monextstate=self.model[u1][i1][moaction][1]
                        u2,i2=self.environment.get_state(monextstate)
                        target=self.get_action_values(monextstate)
                        bonus = kappa * math.sqrt(max(1,time - bi)) 
                        self.Qsa[u1][i1][moaction] += alpha * (moreward + bonus + gamma * target - self.Qsa[u1][i1][moaction])
        
                        if j>planning/3:

                            liss[r][c]=sys.float_info.max      

                       

                
        for i in range(0,self.environment.grid_dimenson[0]):
            for j in range(0,self.environment.grid_dimenson[1]):
                self.policy[i][j]=np.argmax(self.Qsa[i][j])
        return self.Qsa,self.policy 
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
       
width=10
height=10
step=1
list=[]


list.append(3)
list.append(7)
list.append(1)
list.append(7)

list.append(7)
list.append(3)
list.append(1)
list.append(7)

x_start=1
y_start=9
x_end=9
y_end=1
robot1=robot(x_start,y_start,x_end,y_end)

environment=Environment(width,height,list,step)   
reinf=RL(robot1,environment)          
reinf.dyna_qplus(1000,0.9,0.7,0.1,50,0.001)
reinf.plot_policy_grid()
       
                