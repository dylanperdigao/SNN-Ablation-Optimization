# Spiking Neural Network Optimization Strategies for Bank Account Fraud Detection

## Abstract
Spiking Neural Networks (SNNs) have emerged as the next paradigm in Artificial Intelligence, gaining significant popularity due to their high energy eﬃciency when deployed on neuromorphic hardware. However, deploying these networks in real-world scenarios poses optimization challenges, particularly in highly imbalanced datasets, due to the high-dimensional parameter spaces of SNNs. This paper proposes three ablation studies of optimization strategies for SNNs. It provides insights into their parameterization in classification tasks, including the optimization of the membrane decay and firing threshold of the neuron, the number of timesteps for neuronal dynamics simulation, and the surrogate gradient slope required for backpropagation. Our results showed that using a parameter initialization strategy that fixes the parameters while learning only the threshold, with fewer timesteps and a smaller surrogate gradient slope, generally performs better. Furthermore, we use the obtained results to benchmark these optimized SNNs against traditional Machine Learning algorithms, demonstrating a recall of up to 52.45% for online Bank Account Fraud detection.

**Keywords:** Spiking Neural Networks $\cdot$ Neuromorphic Computing $\cdot$ Neuronal Dynamics $\cdot$ Bayesian Optimization $\cdot$ Responsible AI.

![Image](./src/plots/exp_a_lineplot.pdf)
![Image](./src/plots/exp_a_violin.pdf)

![Image](./src/plots/exp_b_lineplot.pdf)
![Image](./src/plots/exp_b_violin.pdf)

![Image](./src/plots/exp_c_lineplot.pdf)
![Image](./src/plots/exp_c_violin.pdf)

![Image](./src/plots/exp_d1_boxplot_recall.pdf)
![Image](./src/plots/exp_d1_boxplot_fairness.pdf)
![Image](./src/plots/exp_d2_radarplot.pdf)
