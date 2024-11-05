
# Learning Heterogeneous Agent Collaboration in Decentralized Multi-Agent Systems via Intrinsic Motivation

## Abstract

Multi-agent Reinforcement Learning (MARL) is emerging as a key framework for various sequential decision-making and control tasks. Unlike their single-agent counterparts, multi-agent systems necessitate successful cooperation among the agents. The real-world deployment of these systems requires decentralized training and execution (DTE), diverse agents, and learning from infrequent environmental rewards. These challenges become more pronounced under partial observability and the lack of prior knowledge about agent heterogeneity. While notable studies use intrinsic motivation (IM) to address reward sparsity or cooperation in decentralized execution settings, those dealing with heterogeneity typically assume centralized training for decentralized execution (CTDE). To overcome these limitations, we propose the CoHet algorithm, which utilizes a novel Graph Neural Network (GNN) based intrinsic motivation to facilitate the learning of heterogeneous agent policies in fully decentralized settings, under the challenges of partial observability and reward sparsity. Evaluation of CoHet in the Multi-agent Particle Environment (MPE) and Vectorized Multi-Agent Simulator (VMAS) benchmarks demonstrates superior performance compared to the state-of-the-art in a range of cooperative multi-agent scenarios. Our research is supplemented by an analysis of the impact of the agent dynamics model on the intrinsic motivation module, insights into the performance of different CoHet variants, and its robustness to an increasing number of heterogeneous agents.

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

The project relies on the following packages --
The packages installed via conda:
-   abseil-cpp
-   bzip2
-   c-ares
-   ca-certificates
-   grpc-cpp
-   grpcio
-   libcxx
-   libffi
-   libprotobuf
-   ncurses
-   openssl
-   pip
-   python
-   re2
-   readline
-   setuptools
-   six
-   sqlite
-   tk
-   wheel
-   xz
-   zlib

And the packages installed via pip:
-   aiosignal
-   amqp
-   appdirs
-   attrs
-   billiard
-   celery
-   certifi
-   charset-normalizer
-   click
-   click-didyoumean
-   click-plugins
-   click-repl
-   cloudpickle
-   contourpy
-   cycler
-   decorator
-   distlib
-   dm-tree
-   docker-pycreds
-   filelock
-   fonttools
-   frozenlist
-   gitdb
-   gitpython
-   gputil
-   gym
-   gym-notices
-   idna
-   imageio
-   imageio-ffmpeg
-   jinja2
-   joblib
-   jsonschema
-   jsonschema-specifications
-   kiwisolver
-   kombu
-   lazy-loader
-   lz4
-   markupsafe
-   matplotlib
-   moviepy
-   mpmath
-   msgpack
-   networkx
-   numpy
-   packaging
-   pandas
-   pathtools
-   pillow
-   platformdirs
-   proglog
-   prompt-toolkit
-   protobuf
-   psutil
-   pyglet
-   pyparsing
-   python-dateutil
-   pytz
-   pywavelets
-   pyyaml
-   ray
-   referencing
-   requests
-   rpds-py
-   scikit-image
-   scikit-learn
-   scipy
-   sentry-sdk
-   setproctitle
-   smmap
-   sympy
-   tabulate
-   tensorboardx
-   threadpoolctl
-   tifffile
-   torch
-   torch-cluster
-   torch-geometric
-   torch-scatter
-   torch-sparse
-   torch-spline-conv
-   tqdm
-   trainutils
-   typing-extensions
-   tzdata
-   urllib3
-   vine
-   virtualenv
-   vmas
-   wandb
-   wcwidth

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

For any inquiries, please contact [Jahir Sadik Monon](https://jahirsadik.github.io/).

## Acknowledgements

We thank the contributors of the Multi-agent Particle Environment (MPE) and Vectorized Multi-Agent Simulator (VMAS) for providing the benchmarks used in this research.
