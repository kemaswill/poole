---
layout: post
title: "Line Search Methods"
---

Line Search Methods
===

Each iteration of a line search method computes a search direction $p_k$, and then decides how far to move along this direction. The iteration is given by

$$x_{k+1}=x_k + \alpha_k p_k\tag{1}$$

The $\alpha_k$ is the step length, and $p_k$ is the search direction, which is a descent direction in most line search algorithms, which means that $p_k^T\nabla f_k < 0$,  because this property guarantees that the function $f$ reduces along the direction. The search direction usually has the following form:

$$p_k=-B_k^{-1}\nabla f_k\tag{2}$$

where $B_k$ is a symmetric and nonsingular matrix. $B_k$ is simply te identity matrix $I$ in steepest descent method, exact Hessian $\nabla^2 f(x_k)$  in Newton's method, and an approximation to the Hessian that is updated at every iteration by means of a low-rank formula.
When $p_k$ is defined by $(2)$ and $B_k$ is positive definite, we have 

$$p_k^T\nabla f_k =-\nabla f_k^TB_k^{-1}\nabla f_k < 0$$

, and therefore $p_k$ is a descent direction.

## Step Length

There is a trade-off in computing step length $\alpha_k$, we would like to get a step length $\alpha_k$ that can substantially reduce the function $f$ without spending too much time, i.e., we want to get a global minimizer of $$\phi(\alpha) = f(x_k + \alpha p_k), \alpha > 0$$, but it's usually too expensive to get the value(see the following figure). So in practice, we prefer to perform an inexact line search to identify a step length that achieves adequate reduction in $f$ at minimal cost.
![enter image description here](https://lh3.googleusercontent.com/83JInNCEuvx_VL-dFjhTr0B0w3lcuaflvPT7STs-DQc=s0 "Screen Shot 2015-08-09 at 下午06.14.40.png")

Line Search Methods use the following strategies to get the adequate step length:
   - A bracketing phase finds an interval containing desierable step lengths
   - A bisection or interpolation phase computes a good step length within the interval
There is different termination conditions for the line search methods, as showed in the following.

### The Wolfe Conditions
A popular inexact line search condition need that $\alpha_k$ should first of all give *sufficient decrease* in the function $f$, as measured by the following inequality:

$$f(x_k + \alpha p_k) \leq f(x_k) + c_1\alpha \nabla f_k^Tp_k$$

for some constant $c_1 \in (0, 1)$. In other words, the reduction in $f$ should be proportional to both the step length $\alpha_k$, and the directional derivative $\nabla f_k^Tp_k \tag{3}$. The $(3)$ inequality is sometimes called the *Armijo condition*. This sufficient decrease condition is showed in the following figure:
![enter image description here](https://lh3.googleusercontent.com/whPcGe44NLU06n5PFZF5QSk5yCL1w_l2KjvsRkwhcyA=s0 "Screen Shot 2015-08-09 at 下午06.14.59.png")
 The right-hand-side of $(3)$, which is a linear function, can be denoted by $l(\alpha)$. The acceptable intervals are showed in the figure. In practice, $c_1$ is chosen to be quite small, say $c_1=10^{-4}$
The sufficient decrease condition itself is not enough to ensure that te algorithms makes reasonable progress, because it's satisfied for all sufficiently small value of $\alpha$, as showed in the above figure. To rule out such short steps, we introduce a second requirement, called the *curvature condition*, which requires $\alpha_k$ to satisfy

$$\nabla f(x_k + \alpha_k p_k)^Tp_k \geq c_2 \nabla f_k^Tp_k$$

 for some constant $c_2 \in (c_1, 1)$, i.e., this conditions ensures that the slope of $\phi(\alpha_k)$ is greater than $c_2$ times the gradient $\phi'(0)$.
This make sense since if te slope $\phi'(\alpha)$ is strongly negative, we can reduce $f$ significantly by moving further along the chosen direction. On the other hand, if the slope is only slightly negative or even positive, it's a sign that we cannot expect much more decrease in $f$ in this direction, so it might make sense to terminate the line search. The curvature condition is illustrated on the following figure
![enter image description here](https://lh3.googleusercontent.com/WdEEesKctLlcmcFaj-NNCvs5Ym4_81VhYvDt8ljpCPc=s0 "Screen Shot 2015-08-09 at 下午06.15.05.png")

The sufficient decrease condition and curvature condition are known collectively as the *Wolfe conditions*:

$$f(x_k + \alpha)kp_k) \leq f(x_k) + c_1 \alpha_k\nabla f_k^Tp_k \tag{4}$$

$$\nabla f(x_k + \alpha_kp_k)^Tp_k \geq c_2\nabla f_k^Tp_K \tag{5}$$

and is illustrated in the following figure:
![enter image description here](https://lh3.googleusercontent.com/cEQAj4MegLh_4tAEQ0flYpfm7FWXif4CKMBTDwfAnG4=s0 "Screen Shot 2015-08-09 at 下午06.15.12.png")

### The Goldstein Conditions
Like the Wolfe conditions, the *Goldstein conditions* also ensure that the step length $\alpha$ achieves sufficient decrease while preventing $\alpha$ from being too small:

$$f(x_k) + (1-c)\alpha_k\nabla f_k^Tp_k \leq f(x_k + \alpha_kp_k) \leq f(x_k) + c\alpha_k \nabla f_k^Tp_k$$

and is showed in the following figure:
![enter image description here](https://lh3.googleusercontent.com/H2J6x6a_C4EFKjY6oCVcDe2y8iaC9W8V8yEQB-xtdPk=s0 "Screen Shot 2015-08-09 at 下午06.15.22.png")

### Sufficient Decrease and BackTracking

As we have mentioned, the sufficient decrease condition itself is not enough to make reasonable progress along the given descent direction. However, we can use a so-called *BackTracking* approach to prevent $\alpha$ from being too small:

   > **BackTracking Line Search**
   > > Choose $\bar{\alpha} > 0,  \rho, c \in (0, 1);$; set $\alpha \gets \bar{\alpha}$
   > > **repeat** until $f(x_k + \alpha p_k) \leq f(x_k) + c\alpha \nabla f_k^Tp_k$
   > > > $\alpha \gets \rho \alpha$
   > > **end **(**repeat**)
   > > Terminate with $\alpha_k = \alpha$

##Rate of Convergence
