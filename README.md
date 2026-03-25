# Deep Symbolic Regression with Reinforcement Learning

**Authors:** Dequatre Julian, L'Herminé Camille, Jelassi Meriem


## Motivation
Discovering concise analytical equations from data is a central challenge for scientific discovery, yet most deep learning methods focus on black-box prediction rather than interpretable structure. Symbolic regression addresses this gap by searching directly over symbolic expressions, but the combinatorial search space becomes extremely large as the number of operators and variables grows. 

Recent work on Deep Symbolic Regression shows that this search can be naturally framed as an RL problem. We study symbolic regression as an RL task in which the agent learns to discover simple, accurate formulas for small benchmark datasets derived from physics, maximizing a reward that balances data fit and complexity.

## Description of the Environment
We define a custom environment in which each episode corresponds to constructing a single mathematical expression for a fixed dataset.

* **State Space:** Encodes (i) the partially constructed expression in prefix notation and (ii) a permutation-invariant embedding of the dataset via a DeepSets-style encoder.
* **Action Space:** Discrete tokens from a grammar, including variables, operators ($+,-,\times,\div,\sin,\cos,\exp,\log$), constants ($e$ , $\pi$) and an end-of-sequence token. The selected token is appended if it preserves syntactic validity.
* **Terminal Reward:** $$R(e)=-\text{NMSE}(e)-\alpha\cdot\text{complexity}(e)$$

where NMSE is the normalized mean squared error and complexity(e) is the expression length.

## Description of the Implemented Agent
* The main agent is a powerful 2-layer LSTM (Hidden Dim: 512) that autoregressively generates mathematical tokens.
* **Vectorized Batched Rollouts**: The generative process bypasses Python loops entirely and constructs 256 structural trees simultaneously via PyTorch matrix operations on the GPU, yielding a ~90x acceleration.
* **Risk-Seeking Policy Gradients (RSPG)**: The RL optimizer trains exclusively on the top 5% elite quantile of expressions to enforce discovery rather than average performance.
* **Top-K Memory**: We employ Teacher Forcing to continuously re-inject the historical top 20 expressions into the gradient buffer to prevent catastrophic forgetting.
* **BFGS Numerical Optimization**: Expressions containing the `const` token are intercepted by a SciPy BFGS solver which calculates the optimal constant (10 iterations) before assigning reward.
* **Physical Vocabulary**: The action space naturally manipulates structural constants (`0.5`, `2.0`, `3.0`, `pi`).

## Discussion and Visualization of Results
* Evaluation is conducted on the extremely complex **Feynman PMLB (Physics)** benchmarks.
* Metrics include test NMSE, exact recovery rate, and average expression length, all exported to CSV logs in real-time.
* Included is a custom AST Visualizer that renders the most complex discovered topologies dynamically using `nx.DiGraph`.

## References
* Petersen et al., "Deep Symbolic Regression (DSR)", ICLR 2021. [arXiv:1912.04871](https://arxiv.org/abs/1912.04871)
* Bastiani et al., "Complexity-Aware DSR with Robust Risk-Seeking Policy Gradients", 2024. [arXiv:2406.06751](https://arxiv.org/abs/2406.06751)
* Udrescu & Tegmark, "AI Feynman: A Physics-Inspired Method for Symbolic Regression". [Science Advances](https://www.science.org/doi/10.1126/sciadv.aay2631)
* Samuel Holt et al., "Deep Generative Symbolic Regression (DGSR)", 2024. [arXiv](https://arxiv.org/abs/2311.14022)
