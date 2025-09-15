# Reinforcement-learning-for-robot-path-planning
This project implements and compares multiple reinforcement learning algorithms for multi-robot path planning in grid-based environments with obstacles. The system enables robots to navigate from start positions to goal positions while avoiding collisions and obstacles using various RL approaches.
# 🤖 Multi-Robot RL Path Planning

A reinforcement learning-based path planning system for multiple robots in a grid world.  
Implements classical and advanced RL algorithms with visualization tools and dynamic environment support.

---

## 🔑 Key Features

- **Multiple RL Algorithms**: Value Iteration, Q-Learning, Dyna-Q, Dyna-Q+
- **Multi-Robot Coordination**: Supports multiple robots with collision avoidance
- **Grid World Environment**: Customizable grid sizes and obstacles
- **Visualization Tools**: Animated robot paths and policy grid visualization
- **Dynamic Environment Support**: Dyna-Q+ handles changing environments

---

## 🧠 Algorithms Implemented

- `value_iteration.py` – Model-based RL using dynamic programming  
- `Q_learning.py` – Model-free, off-policy TD learning  
- `dyna_q.py` – Combines real experience with model-based planning  
- `dyna_qplus.py` – Enhanced Dyna-Q for dynamic environments

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.7+
- NumPy
- Matplotlib

### 💾 Installation

```bash
git clone <your-repo-url>
cd multi-robot-rl-path-planning
pip install numpy matplotlib
▶️ Usage
Run any of the algorithms to see path planning in action:

bash
Copy code
python value_iteration.py
python Q_learning.py
python dyna_q.py
python dyna_qplus.py
Each script will:

Initialize the grid world with obstacles

Train the RL agent(s)

Display the learned policy grid

Animate the robot(s)' path

Show final paths for all agents

🗂️ File Structure
bash
Copy code
├── value_iteration.py     # Value Iteration algorithm
├── Q_learning.py          # Q-Learning algorithm  
├── dyna_q.py              # Dyna-Q algorithm
├── dyna_qplus.py          # Dyna-Q+ algorithm
└── README.md              # This file
⚙️ Configuration
🔧 Environment Setup
Edit these parameters in each script:

python
Copy code
width = 10                # Grid width
height = 10               # Grid height
step = 1                  # Movement step size
list = []                 # Obstacle list [x, y, width, height, ...]
🤖 Robot Configuration
python
Copy code
# Define robots with start and end positions
robot1 = robot(x_start, y_start, x_end, y_end)
agents = [robot1, robot2, ...]  # Multiple agents
📊 Results
Each algorithm provides:

Policy grids with optimal actions

Animated robot paths

Collision-free multi-robot coordination

Efficient obstacle-avoiding trajectories

🎨 Visualization
Grid world with gray obstacles

Start positions: 🟢 ‘S’

Goal positions: 🔴 ‘G’

Policy directions with arrows

Animated multi-robot movement

Color-coded robot paths

🔧 Customization
You can easily:

Change grid dimensions and obstacle layout

Adjust learning parameters: α (alpha), γ (gamma), ε (epsilon)

Modify rewards

Add more robots

Test new obstacle configurations

📈 Performance Summary
Value Iteration: Fast, optimal with known models

Q-Learning: Good for unknown environments

Dyna-Q: Efficient with both learning and planning

Dyna-Q+: Adapts to dynamic, changing environments

🤝 Contributing
Contributions are welcome! You can:

Add new RL algorithms

Improve visualization

Optimize code performance

Enhance environment features

📄 License
This project is open source and available under the MIT License.
