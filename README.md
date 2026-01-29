# Adam (Adaptive Moment Estimator) Optimizer

## What is Adam?

Adam (Adaptive Moment Estimation) is an algorithm for first-order gradient based optimization of stochastic objective functions based on adaptive estimates of lower-order moments.

Some of Adam’s advantages are:

1. The magnitudes of parameter updates are invariant to rescaling of the gradient 
2. Its stepsizes are approximately bounded by the stepsize hyperparameter 
3. It does not require a stationary objective (i.e., it will still converge if f(ϑ) changes) 
4. It works with sparse gradients
5. It naturally performs a form of step size annealing

[Pseudocode figure from Kingma and Ba 2015](https://github.com/UMT-CSCI-557/optimizers/blob/main/Pseudocode.png)

<br>

## Moments

First moment ($$𝑚_𝑡$$): the mean of the gradient

$$𝑚_𝑡= {\beta _1} ⋅ m_{𝑡−1}+(1 − {\beta _1})⋅ 𝑔_𝑡$$

Second moment ($$𝑣_𝑡$$): the raw, uncentered variance of the gradient

$$𝑣_𝑡= 𝛽_2 ⋅ 𝑣_{𝑡−1} + (1 − {\beta_2})⋅ 𝑔_𝑡^2$$

βs (decay rates) give more weight to recent gradients

However, these moments are **BIASED** early on because they are initialized at 0

Hence, we have to correct the bias

$$\hat{𝑚_𝑡}= 𝑚_𝑡∕(1 − {\beta_1^𝑡})$$ 
$$\hat{𝑣_𝑡}= 𝑣_𝑡∕(1 − {\beta_2^𝑡})$$ 

But where does this bias correction come from?

First, we rewrite the moment as a function of all previous gradients.

$$𝑣_𝑡 = (1 − {\beta _2}) \sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} ⋅ 𝑔_𝑖^2$$

We are trying to find the *expected* value of the gradient.

$$𝔼[𝑣_𝑡] = 𝔼[(1 − {\beta _2}) \sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} ⋅ 𝑔_𝑖^2]$$

Next, we assume that the gradient is constant.

$$𝔼[𝑣_𝑡] = 𝔼[𝑔_𝑖^2] ⋅ (1 − {\beta _2}) \sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} + \zeta$$

What the paper fails to make explicit is that the summation here is actually a **geometric series** (i.e., when summing those values, sequential terms are related to each other by a specific ratio, in this case, $\frac{1}{\beta}$). These two terms are equivalent:

$$\sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} = \frac{1 − {\beta_2^𝑡}}{1 − {\beta_2}}$$

Therefore, we can sub in the ratio $\frac{1 − {\beta_2^𝑡}}{1 − {\beta_2}}$ for the summation.

$$𝔼[𝑣_𝑡] = 𝔼[𝑔_𝑖^2] ⋅ (1 − {\beta _2}) \frac{1 − {\beta_2^𝑡}}{1 − {\beta_2}} + \zeta$$

And then we can arrive at the final equation shown in the paper by cancelling out the $1 - {\beta_2}$:

$$𝔼[𝑣_𝑡] = 𝔼[𝑔_𝑖^2] ⋅ (1 − {\beta_2^𝑡}) + \zeta$$

So, because of this extra $1 - {\beta_2^t}$ that appears in this equation, that's why we divide $m_t$ and $v_t$ by $1 - {\beta_2^t}$.

<br>

## Step Size

The change in our step size at time t is:

$${\Delta_𝑡} = \alpha ⋅ \frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}}}$$

, where 𝛼 is a “maximum” step size parameter. If you want to take N steps to the optimum that is D distance away, then $\alpha \approx \frac{𝐷}{𝑁}$

The effective step sizes are approximately bound to the step size hyperparameter. The step size will not grow too large except in the case of severe sparsity (when a gradient has been zero at all timesteps except at the current timestep), such as in this case:

$$|\Delta_t| \leq \alpha  ⋅ \frac{1 - \beta_1}{\sqrt{1 - \beta_2}}, \qquad \text{if }  (1 - \beta_1) \gt \sqrt{1 - \beta_2}$$   

Otherwise, when the gradient isn't sparse:  
  
<p align="center">$$|\Delta_t| \leq \alpha, \qquad \text{if }  (1 - \beta_1) \leq \sqrt{1 - \beta_2}$$          



$\frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}}}$ is considered to be a signal-to-noise ratio (SNR). When SNR is small, the step size decreases. SNR typically decreases when approaching an optimum, where we want smaller effective steps.

Since the final step equation divides the estimated mean by the estimated variance (1st moment / 2nd moment), any gradient scaling cancels out.​

$$\frac{𝑐 ⋅ \hat{𝑚_𝑡}}{\sqrt{(𝑐^2  ⋅ \hat{𝑣_𝑡}}}= \frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}}}$$

This means that, no matter what scale your inputs are, Adam will take the same step size – only $\alpha$ affects the step size.

Now we can finally update our parameter values!

$${\theta_t} = {\theta_{𝑡 − 1}} − {\alpha _t}  ⋅ \frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}} + \epsilon}$$

(𝜀 is there to prevent dividing by 0.)

