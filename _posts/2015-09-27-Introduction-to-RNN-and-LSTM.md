---
layout: post
title: Introduction to LSTM
---

## Recurrent Neural Networks

Traditional feedforward networks doesn't have connections that form cycles, while recurrent neural networks relax such constriction. Below is a simple RNN which contains a single, self connected hidden layer:
![enter image description here](https://lh3.googleusercontent.com/-K3o_r3uD1dg/VfGYIjbnfCI/AAAAAAAAAKw/IvLbYoqCqQU/s0/Screen+Shot+2015-09-10+at+%25E4%25B8%258A%25E5%258D%258807.47.03.png "Screen Shot 2015-09-10 at 上午07.47.03.png")

While the difference between RNN and MLP seems trivial, the cycles connections allow RNN to map the entire history to the output, so it's much more suitable for sequence learning. Indeed, the equivalent result to the universal approximation theory of MLPs is that an RNN with a sufficient number of hidden units can approximate any measurable sequence-to-sequence mapping to arbitrary accuracy.

### Forward Pass

The forward pass of RNN is very similar with that of MLPs, except that the input arrives at the hidden layer from both the current input and the hidden layer activations from the previous timestamp. For a RNN with $I$ input units, $H$ hidden units, $K$ output units. Let $x_i^t$ be the value of input $i$ at time $t$, and let $a_j^t$ and $b_j^t$ be respectively the network input to unit $j$ at time $t$ and the activation of unit $j$ at time t. For the hidden units we haveL

$$a_h^t = \sum_{i=1}^I w_{ih}x_i^t + \sum_{h'=1}^Hw_{h'h}b_{h'}^{t-1}\tag{1}$$

Then a nonlinear, differential activation functions can be applied:

$$b_h^t = \theta_h (a_h^t)\tag{2}$$

The complete sequence of hidden activations can be calculated by starting at $t=1$ and recursively applying $(1)$ and $(2)$, incrementing $t$ at each step. Note that this requires initial values $b_i^0$ to be chosen from hidden units, corresponding to the network's state before it receives any information from the data sequence. We can set such initial values to zero, however it's better to set them as nonzero.

The output activation can be calculated at the same time as hidden activations:

$$a_{k}^t = \sum_{h=1}^H w_{hk} v_h^t$$

### Backward Pass

There are two algorithms for calculating weight derivatives for RNNs: real time recurrent learning(RTRL) and back propagation through time(BPTT), here we focus on BPTT.

Like back propagation for MLPs, the BPTT consists of a repeated application of chain rule. Therefore:

$$\delta_h^t=\theta'(a_h^t)(\sum_{k=1}^K\delta_k^t w_{hk} + \sum_{h'=1}^H\delta_{h'}^{t+1}w_{hh'})\tag{3}$$

where

$$\delta_j^t = \frac{\partial \mathcal{L}}{\partial a_j^t}$$.

The complete sequence of $\delta$ terms can be calculated by starting at $t=T$ and recursively applying $(3)$, decrementing $t$ at each step.(Note that $\delta_j^{T+1} = 0, \forall j$). Note that the same weights are reused at each timestamp, we sum over the whole sequence to get the derivatives w.r.t. the network weights:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{t=1}^T\frac{\partial \mathcal{L}}{\partial a_j^t}\frac{\partial a_j^t}{\partial w_{ij}}=\sum_{t=1}^T\delta_j^tb_i^t$$

### Unfolding

A useful way to visualize RNNs is to unfold the RNNs along time to a MLP without any cyclic connections. The unfolded MLP has a layer for each "time step" in the sequence, as if the "time step" is the index of the layer. 

![enter image description here](https://lh3.googleusercontent.com/-OztP080zQVQ/VfGkVSXkanI/AAAAAAAAALI/yDy0K_N9F0k/s0/Screen+Shot+2015-09-10+at+%25E4%25B8%258A%25E5%258D%258807.47.13.png "Screen Shot 2015-09-10 at 上午07.47.13.png")

## Long Short-Term Memory

An important benefit or RNNs compared with traditional feedforward networks is their ability to use contextual(history) information when mapping between input and output sequences. However, due to vanishing gradient problem, the range of context that can be in practice accessed by standard RNN architectures is quite limited.

### Vanishing Gradient Problem

![enter image description here](https://lh3.googleusercontent.com/-Yqih-Z-1_D0/VfGkaG4X6lI/AAAAAAAAALU/vppLm1KUzMA/s0/Screen+Shot+2015-09-10+at+%25E4%25B8%258A%25E5%258D%258808.39.31.png "Screen Shot 2015-09-10 at 上午08.39.31.png")

### Network Architecture

The LSTM consists of a set of recurrently connected subnets, known as memory blocks. Each such memory block contains one or more self-connected memory cells and three multiplicative units: the input, output and forget gates.

The following figure shows an LSTM memory block with a single cell. An LSTM network is the same as standard RNN, excepts that the summation units in te hidden layer are replaced by memory blocks.

![enter image description here](https://lh3.googleusercontent.com/-JaB0UjWnIXQ/VfGkqv0BHyI/AAAAAAAAALg/_O9ctImxi1g/s0/Screen+Shot+2015-09-09+at+%25E4%25B8%258A%25E5%258D%258808.36.30.png "Screen Shot 2015-09-09 at 上午08.36.30.png")

The multiplicative gates allow LSTM memory cells to store and access information over long periods of time, thereby mitigating the vanishing gradient problem.

![enter image description here](https://lh3.googleusercontent.com/-X2BPtS32l4E/VfGl66Xpt0I/AAAAAAAAAL0/FAfgtScvyFk/s0/Screen+Shot+2015-09-10+at+%25E4%25B8%258A%25E5%258D%258808.46.32.png "Screen Shot 2015-09-10 at 上午08.46.32.png")

### Forward Pass

$w_{ij}$ is the weight of connection fro unit $i$ to unit $j$
$I, K,H$ is the number of inputs, outputs and memory cells in hidden layer. $C$ is the number of memory cells in a memory block.
$w_{c\iota}, w_{c\phi}, w_{c\omega}$ is the peephole weights from cell $c$ to the input, forget and output gates.
Note that only the cell outputs $b_c^t$ are connected to the other cells in the hidden layer. The other LSTM activations, such as the states, the cell inputs, or the gate activations, are only visible within the block. We use $h$ to refer to cell outputs from other blocks in the hidden layer.

#### Input Gates

$$a_{\iota}^t = \sum_{i=1}^Iw_{i\iota}x_i^t + \sum_{h=1}^Hw_{h\iota}b_h^{t-1} + \sum_{c=1}^Cw_{c\iota}s_c^{t-1}$$

$$b_{\iota}^t = f(a_\iota^t)$$

#### Forget Gates

$$a_{\phi}^t = \sum_{i=1}^Iw_{i\phi}x_i^t + \sum_{h=1}^Hw_{h\phi}b_h^{t-1} + \sum_{c=1}^Cw_{c\phi}s_c^{t-1}$$

$$b_{\phi}^t = f(a_\phi^t)$$

#### Output Gates

$$a_{\omega}^t = \sum_{i=1}^Iw_{i\omega}x_i^t + \sum_{h=1}^Hw_{h\omega}b_h^{t-1} + \sum_{c=1}^Cw_{c\omega}s_c^{t-1}$$

$$b_{\omega}^t = f(a_\omega^t)$$

#### Cells

$$a_c^t=\sum_{i=1}^Iw_{ic}x_i^t + \sum_{h=1}^Hw_{hc}b_h^{t-1}$$

$$s_c^t = b_{\phi}^ts_c^{t-1}+b_{\iota}^tg(a_c^t)$$

#### Cell Outputs

$$b_c^t = b_{\omega}^th(s_c^t)$$

### Backward Pass

$$\epsilon_c^t = \frac{\partial \mathcal{L}}{\partial b_c^t}, \epsilon_s^t=\frac{\partial \mathcal{L}}{\partial s_c^t}$$

#### Cell Outputs

$$\epsilon_c^t=\sum_{k=1}^Kw_ck\delta_k^t + \sum_{g=1}^Gw_{cg}\delta_g^{t+1}$$

#### Output Gates

$$\delta_w^t=f'(a_w^t)\sum_{c=1}^Ch(s_c^t)\epsilon_c^t$$

#### States

$$\epsilon_s^t=b_{\omega}^th'(s_c^t)\epsilon_c^t + b_{\phi}^{t+1}\epsilon_s^{t+1} + w_{c\iota}\delta_{\iota}^{t+1}+ w_{c\phi}\delta_{\phi}^{t+1} + w_{c\omega}\delta_{\omega}^t$$

#### Cells

$$\delta_c^t=b_{\iota}^tg'(a_c^t)\epsilon_s^t$$

#### Forget Gates

$$\delta_{\phi}^t=f'(a_{\phi}^t)\sum_{c=1}^Cs_c^{t-1}\epsilon_s^t$$

#### Input Gates

$$\delta_{\iota}^t=f'(a_{\iota}^t)\sum_{c=1}^Cg(a_c^t)\epsilon_s^t$$

### How Does LSTM Works?

Here we explain how does LSTM works, e.g., how does the output, cell state and cell output, and 3 kinds of gates change when reading the sequence. 

The input layer has 3 input nodes, corresponding to $a$, $b$, $S$ respectively. The hidden layer contains 1 memory block, which has 1 memory cell within it. The output layer has 3 output, corresponding to $a$, $b$, $T$ respectively. The LSTM is trained on a Context Free Language, e.g., all training sequences have the pattern $a^nb^n$. After the model is trained, we feed a sequence $Saaaaabbbbb$ to the LSTM to verify whether it can recognize this sequence as a valid CFL(Here $S$ means $start$, and $T$ means $end$). Following figures shows how does the network output, cell state and cell output, activation of all gates changes when reading this sequence:

![enter image description here](https://lh3.googleusercontent.com/-eMWUC5k-j7E/VfjrEzK036I/AAAAAAAAAMs/NBL_SFpcnU8/s0/Screen+Shot+2015-09-15+at+%25E4%25B8%258B%25E5%258D%258809.06.47.png "Screen Shot 2015-09-15 at 下午09.06.47.png")

 1. Network Output. The output shows the possible next character based on the history, say that the current sequence is $Saa$, then the next character can be either $a$ or $b$, the $a$ corresponds to all $a^nb^n$ with $n \geq 3$, and the $b$ corresponds to $aabb$
 2. The state of cell is just a count of $n$, and will increase when reading the character $a$, then decreases when reading the character $b$.
 3. The output gate stay zero before the last $b$ is seen, and then jump to 1.0 after seeing the last $b$, then the network output $T$, which means the end of the sequence.


[1]. Supervised Sequence Labelling with Recurrent Neural Networks. Alex Graves.

[2]. A Tutorial on Training RNNs, covering BPPT, RTRL, EKF and the "echo state network" approach.

[3]. LSTM Recurrent Networks Learning Simple Context Free and Context Sensitive Languages. Felix A. Gers, Jurgen Schmidhuber.