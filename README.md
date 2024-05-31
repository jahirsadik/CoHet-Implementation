
# Towards Heterogeneous Agent Cooperation in Decentralized Multi-Agent Reinforcement Learning

## Abstract

Multi-agent Reinforcement Learning (MARL) is gaining significance as a key framework for various sequential decision-making and control tasks. Unlike their single-agent counterparts, multi-agent systems necessitate successful cooperation among the agents. The deployment of these systems in real-world scenarios often requires decentralized training, heterogeneous agents, and learning from sparse environmental rewards. These challenges are more acute under partial observability and the lack of prior knowledge about agent heterogeneity. While notable studies use intrinsic motivation (IM) to address reward sparsity in decentralized settings, those dealing with heterogeneity typically assume centralized training, parameter sharing, and agent indexing. To address these issues, we propose the CoHet algorithm, which utilizes a novel Graph Neural Network (GNN) based intrinsic motivation to facilitate the learning of heterogeneous agent policies in decentralized settings under conditions of partial observability and reward sparsity. Evaluation of CoHet in the Multi-agent Particle Environment (MPE) and Vectorized Multi-Agent Simulator (VMAS) benchmarks demonstrates that it outperforms the state-of-the-art in a wide range of cooperative multi-agent scenarios. Our research is supplemented by an analysis of the impact of our agent dynamics model on the intrinsic motivation module, how the different variants of CoHet perform, and its robustness to an increasing number of heterogeneous agents.

## Installation

### Requirements

To install the necessary dependencies, you can use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Alternatively, you can set up the environment using the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate cohet-env
```

### Dependencies

The project relies on the following packages:

- numpy==1.23.5
- vmas
- wandb
- torch
- torch_geometric
- torch_scatter
- torch_sparse
- torch_cluster
- torch_spline_conv

Please ensure all the dependencies are installed before running the code.

## Usage

To run the CoHet algorithm, follow the steps below:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/jahirsadik/CoHet-Implementation.git
    cd CoHet-Implementation
    ```

2. **Set up the environment**:
    Use either `requirements.txt` or `environment.yml` as mentioned in the Installation section.

3. **Run the training script**:
    ```bash
    python train/train_navigation.py # to run the training on VMAS-NAVIGATION
    ```

4. **Monitor training progress**:
    Training progress can be monitored using Weights & Biases (wandb). Make sure to set up a wandb account and login using:
    ```bash
    wandb login
    ```
    
## Results

Our evaluations demonstrate that CoHet outperforms state-of-the-art methods in various cooperative multi-agent scenarios. 


## Contributing

We welcome contributions to the CoHet project! Please open an issue or submit a pull request if you have any improvements or suggestions.


## Contact

For any inquiries, please contact [Jahir Sadik Monon](mailto:jahirsadikmonon@gmail.com).

## Acknowledgements

We thank the contributors of the Multi-agent Particle Environment (MPE) and Vectorized Multi-Agent Simulator (VMAS) for providing the benchmarks used in this research.
