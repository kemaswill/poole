---
layout: post
title: "Matrix Factorization"
---


## Matrix Factorization

In Recommender System, **Matrix Factorization** maps both the users and items to a joint latent factor space of dimension $k$, such that the user-item interaction can be modeled as the inner product in this space. That is to say, we map each user $i$ to a vector $p_i\in \mathbb{R}^k$, and each item $j$ to a vector $q_j \in \mathbb{R}^k$. For movie recommendation, each dimension of the latent factor space can be explained as a topic, say comedy v.s. drama, or other features such as amount of action, orientation to children and so on.
Given the latent vector for user $u$ and item $i$, we can predict the interaction between them as 

$$\hat{r_{ui}}=q_i^Tp_u\tag{1}$$

The major challenge is to compute the latent vector for each user and item. This is quite similar with **Singular Value Decomposition(SVD)**, which .... However, the matrix $M$ is needed to be complete when using SVD to decompose it. One method is to rely on imputation to fill in missing ratings to make the matrix dense. However, this will significantly increase the amount of data, and the inaccurate imputation might distort the data.
Matrix Factorization is a method which focus only on the observed ratings only, while avoid overfitting by regularization. Here is the cost function for matrix factorization

$$\min_{q^*, p^*}\sum_{(u, i) \in \kappa}(r_ui - q_i^Tp_u)^2 + \lambda (||q_i||^2 + ||p_u||^2)\tag{2}$$

Here $\kappa$ is the set of $(u, i)$pairs for which $r_{ui}$ is known in the training set.

### Optimization by SGD

The above cost function $(2)$ works as following

> Stochastic Gradient Descent for Matrix Factorization
> 
> For each rating $r_{ui}$ in the training set
> 
> $e_{ui} := r_{ui} - q_i^T p_u$
> 
> $q_i \gets q_i + \gamma(e_{ui} p_u -\lambda q_i)$
> 
> $p_u \gets p_u + \gamma (e_{ui}q_i -\lambda p_u)$

### Adding Biases

The $(2)$ only interpret the rating $r_{ui}$ as an interaction between the user $u$ and item $i$, but in the fact,  the rating values can also due to effects associated with either users or items. For example, in the recommender system, some user tend to give higher rating than others, or some items is in general better than others. To consider all such effects, we can add a bias term to the $(1)$

$$\hat{r_{ui}} = \mu + b_i + b_u + q_i^Tp_u\tag{3}$$

 and the corresponding cost function is 
 
$$\min_{p^*,q^*, b^*}\sum_{(u, i)\in \kappa}(r_{ui}-\mu-b_u-b_i-p_u^Tq_i)^2 + \lambda(||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)\tag{4}$$

## Factorization Machines

Original Matrix Factorization use only the rating information. What if we can get more features about the user and item? Such as the gender and age information of user, or the category or sale information about the item. Koren has mentioned that we can also use matrix factorization when adding more informations. For example, if we also have the implicit feedback such as the purchase or browsing history, as well as some user attributes.
We denote $N(u)$ as the sets of items for which user $u$ has expressed an implicit feedback,  where each item $i$ is associated with $x_i \in \mathbb{R}^f$. So a user who showed a preference for items in $N(u)$ is characterized by 

$$|N(u)|^{-0.5}\sum_{i\in N(u)}x_i\tag{4}$$

Denote $A(u)$ as the set of attributes of a user $u$, and a factor vector $y_a \in \mathbb{R}^f$ corresponds to each attribu to describe a user through the set of user-associated attributes:

$$\sum_{a\in A(u)}y^a\tag{5}$$

Then the predicted rating can be modeled by

$$\hat{r}_{ui}=\mu + b_i + b_u + q_i^T[p_u + |N(u)|^{-0.5}\sum_{i\in N(u)}x_i + \sum_{a\in A(u)}y^a]\tag{6}$$

Although Matrix Factorization can model such kind of implicit feedback and user or item attribute features, **Factorization Machine** can handle such kind of features more directly. Assume that the user $u$ and item $i$ have feature vectors$f_u$ and $g_i$, we can formulate the following regression cost function:

$$\min_w\sum_{u, i \in R}(R_{u, i} - w^T\begin{bmatrix}f_u \\ g_i\end{bmatrix})\tag{7}$$

The following cost function is a only linear combination of user and item features, which doesn't consider the interaction between them. We can use the degree-2 polynomial mapping to hander such interaction:

$$\min_{w_{t, s}\forall t, s}\sum_{u, v \in R}(r_{u, i}-\sum_{t'=1}^U\sum_{s'=1}^Vw_{t', s'}(f_u)_{t'}(g_i)_{s'})^2\tag{8}$$

This is equivalent to 

$$\min_W\sum_{u, i\in R}(r_{u, i}-f_u^TWg_i)^2\tag{9}$$

However, this setting fails for extreme sparse features. 
Consider the most extreme situation where the $f_u$ and $g_i$ is the user ID and item ID features, then the optimal solution is 

$$w_{u, i}=\begin{cases}r_{u, i}  & \text{if } u, i \in R \\ 0, & \text{if } u, i \notin r\end{cases}\tag{10}$$

So that we can never predict 

$$r_{u, i}, u, i \notin R$$

The reason is that overfitting occurs, since the number of variables is much more than the number of instances:

$$\text{#variables} = mn \gg \text{#instances} = |R|$$

We can avoid this by letting

$$W \approx P^TQ$$

, where $P$ and $Q$ are both low-rank matrices. So it becomes matrix factorization.
So we can reformulate $(9)$ as 

$$\min_{u, i \in R}(R_{u, i} - f_u^TP^TQg_i)^2$$

We can think 

$$Pf_u \text{ and }Qg_i$$

 as the latent representation of user $u$ and item $i$ in the latent space respectively. This is **Factorization Machine**.

## Field-Aware Factorization Machine

Factorization Machine can effectively model the interaction between user and item, as well as the user side and item side features. But what if there are more than 3 dimension? For example, in the CTR prediction for computational advertising, we may have User, Advertisement as well as Publisher. There is interaction between the User and Advertisement, as well as interaction between User and Publisher. The Field-Aware Factorization Machine can handle all such interactions.
The formulation of FFM is

$$\min_w \sum_{i=1}^L(log(1 + exp(-y_I \phi(w, x_i)))  + \frac{\lambda}{2}||w||^2$$

where

$$\phi(w, x) = \sum_{j_1, j_2 \in C_2}\langle w_{j_1, f_2}, w_{j_2, f_1} \rangle x_{j_1}x_{j_2}$$

where $f_1, f_2$ are respectively the field of $j_1$ and $j_2$ , and $w_{j_1, f_2}$ and $w_{j_2, f_1}$ are two vectors with length $k$ . Here the field means the dimension, for the recommender system, typically there are 2 dimensions: User and Item. For CTR prediction, there are typically 3 dimensions: User, Advertisement and Publisher. But actually we can build one dimension for each ID features, or even category features. 
YuChin Juan has showed the formulation from linear model, to degree-2 polynomial model, to factorization machine and finally field-aware factorization machine models.

The formulation of **linear model** is

$$\phi(w, x) = w^Tx = \sum_{j\in C_1}w_jx_J$$

where $C_1$ is the non-zero elements in $x$.
The formulation of **Poly-2** is 

$$\phi(w, x) = \sum_{j_1, j_2 \in C_2}w_{j_1, j_2}x_{j_1}x_{j_2}$$

where $C_2$ is the 2-combination of non-zero elements in $x$.

The formulation of **factorization machine** is 

$$\phi(w, x) = \sum_{j_1, j_2 \in C_2}\langle w_{j_1}, w_{j_2}x_{j_1}x_{j_2}\rangle$$

where $w_{j_1}$ and 

$w_{j_2}$ are two vectors with length 

$k$

 , and $k$ is a used-defined parameter.

The formulation of **Field-Aware Factorization Machine** is 

$$\phi(w, x)=\sum_{j_1, j_2 \in C_2}\langle w_{j_1, f_2}, w_{j_2, f_1}\rangle x_{j_1}x_{j_2}$$

where $f_1$ and $f_2$ are respectively the fields of $j_1$ and $j_2$.
Here is a concrete example, say there is a sample:

|User(Us)|Movie(Mo)|Gender(Ge)|Price(Pr)|
|--|--|--|--|
|YuChin(YC)|3Idiots(3I)|Comedy, Drama(Co, Dr)|$9.99|

The $\phi(w, x)$ in FFM is 

$$\langle w_{Us-Yu ,Mo} ,w_{ Mo-3I, Us}\rangle x_{Us-Yu} x_{Mo-3I}+ \langle w_{Us-Yu ,Ge} ,w_{ Ge-Co, Us}\rangle x_{Us-Yu} x_{Ge-Co} + \langle w_{ Us-Yu,Ge}, w_{Ge-Dr , Us}\rangle x_{Us-Yu} x_{Ge-Dr}  + \langle w_{ Us-Yu, Pr}, w_{ Pr, Us}\rangle x_{Us-Yu} x_{Pr} $$
$$+ \langle w_{Mo-3I , Ge} ,w_{ Ge-Co, Mo}\rangle x_{Mo-3I} x_{Ge-Co}+ \langle w_{Mo-3I ,Ge} ,w_{Ge-Dr , Mo}\rangle x_{Mo-3I} x_{Ge-Dr}+ \langle w_{Mo-3I ,Pr} ,w_{ Pr, Mo}\rangle x_{Mo-3I} x_{Pr} $$
$$+  \langle w_{Ge-Co ,Ge} ,w_{Ge-Dr , Ge}\rangle x_{Ge-Co} x_{Ge-Dr}+ \langle w_{Ge-Co ,Pr} ,w_{ Pr, Ge}\rangle x_{Ge-Co} x_{Pr}$$
$$ + \langle w_{Ge-Dr ,Pr} ,w_{Pr , Ge}\rangle x_{Ge-Dr} x_{Pr}$$

Reference:

[1]. Matrix Factorization Techniques for Recommender System. Y. Koren, et. at.

[2]. Matrix Factorization and Factorization Machine for Recommender Systems[Slides]. Chih-Jen Lin.

[3]. Factorization Machines. Steffen Rendle.

[4]. 3 Idiots' Approach for Display Advertising Challenge[Slides]. Yu-Chin Juan, Yong Zhuang, and Wei-Sheng Chin.

[5]. Field-aware Factorization Machine[Slides]. Yu-Chin Juan, Yong Zhuang, and Wei-Sheng Chin.

[6]. Pairwise interaction tensor factorization for personalized tag recommendation. Steffen Rendle, Lars Schmidt-Thieme.