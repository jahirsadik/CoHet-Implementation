# Abstract
Cooperative Multi-Agent Reinforcement Learning (MARL) is gaining
significance as a key framework for various sequential decision-making
and control tasks. The successful deployment of these systems is often
hindered by challenges such as reward sparsity and agent heterogene-
ity. These challenges are more acute under partial observability and
decentralized training. To solely address reward sparsity, intrinsic
motivation has been used extensively. However, the notable studies
addressing both challenges assume centralized training and parameter
sharing. In this paper, we introduce the CoHet algorithm, designed to
tackle both reward sparsity and agent heterogeneity in decentralized
training settings under partial observability. We empirically evaluate
the algorithm in the Multi-agent Particle Environment (MPE) and
Vectorized Multi-Agent Simulator, and demonstrate that it outper-
forms the state-of-the-art in a range of sparse cooperative tasks that
require agent heterogeneity. Our research is supplemented by an ex-
amination of the agent dynamics model and its impact on the intrinsic
reward calculation module, along with an analysis of how the intrinsic
reward module helps in dealing with reward sparsity.
