# Abstract
Multi-agent Reinforcement Learning (MARL) is gaining significance as a key
framework for various sequential decision-making and control tasks. Unlike their
single-agent counterparts, multi-agent systems necessitate successful cooperation
among the agents. The deployment of these systems in real-world scenarios often
requires decentralized training, heterogeneous agents, and learning from sparse
environmental rewards. These challenges are more acute under partial observability 
and the lack of prior knowledge about agent heterogeneity. While notable studies 
use intrinsic motivation (IM) to address reward sparsity in decentralized settings, 
those dealing with heterogeneity typically assume centralized training, parameter 
sharing, and agent indexing. To address these issues, we propose the CoHet algorithm, 
capable of learning heterogeneous agent policies in a decentralized manner under 
conditions of reward sparsity and partial observability. Evaluation of CoHet in the 
Multi-agent Particle Environment (MPE) and Vectorized Multi-Agent Simulator (VMAS) 
demonstrate that it outperforms the state-of-the-art in a wide range of cooperative 
multi-agent scenarios. Our research is supplemented by an analysis of the impact of
our agent dynamics model on the intrinsic motivation (IM) module, how the two different
variants of CoHet perform, and its scalability with an increasing number of heterogeneous agents.
