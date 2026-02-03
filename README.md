# Adam (Adaptive Moment Estimator) Optimizer

## What is Adam?

Adam (Adaptive Moment Estimation) is an algorithm for first-order gradient based optimization of stochastic objective functions based on adaptive estimates of lower-order moments.

Some of Adam’s advantages are:

1. The magnitudes of parameter updates are **invariant to rescaling of the gradient** 
2. Its **stepsizes are approximately bounded** by the stepsize hyperparameter 
3. It does not require a **stationary objective** (i.e., it will still converge if $f(\theta)$ changes) 
4. It works with **sparse gradients**
5. It naturally performs a form of **step size annealing**

<img width="857" height="536" alt="Pseudocode" src="https://github.com/user-attachments/assets/2747908d-794b-4833-ae26-cd9a42447748" />

<br>

# Moments

The first moment ($$𝑚_𝑡$$) is the **mean** of the gradient. It Helps Adam build speed towards the optimum, like a ball rolling down a hill.

$$𝑚_𝑡= {\beta _1} ⋅ m_{𝑡−1}+(1 − {\beta _1})⋅ 𝑔_𝑡$$

<img width="1497" height="318" alt="image" src="https://github.com/user-attachments/assets/3513a0dc-737e-43ba-a06b-489f1064cf2c" />


The second moment ($$𝑣_𝑡$$) is the **raw, uncentered variance** of the gradient. It helps Adam normalize its step sizes. If you look at the loss surface below, it is very oblong; other optimizers have a tendency to go down the steepest gradient (in this diagram, the y-axis) before going down the less steep gradient (in this diagram, the x-axis), which makes them slow to converge. The second moment allows for more efficient navigation to the optimum.

$$𝑣_𝑡= 𝛽_2 ⋅ 𝑣_{𝑡−1} + (1 − {\beta_2})⋅ 𝑔_𝑡^2$$

<img width="1174" height="388" alt="image" src="https://github.com/user-attachments/assets/0fe0afe2-7d66-403a-b7de-d3dbe29195a7" />

What is this uncentered variance that the second moment is using? It’s the 'variance' of the gradient. Think of it as encoding how the gradient is changing locally, like the curvature of a function.


## Decay Rate

The βs (decay rates) give more weight to recent gradients. Typically, $\beta_1$ is set to 0.99, while $\beta_2$ is set to 0.999. We can see more clearly how the recency weighting works if we expand what's happening during each time step. We start here:

$$𝑣_1 = 𝛽_2 ⋅ 𝑣_{0} + (1 − {\beta_2})⋅ 𝑔_1^2$$

Then when we move on to the next timestep, now we are here:

$$𝑣_2 = 𝛽_2 ⋅ 𝑣_{1} + (1 − {\beta_2})⋅ 𝑔_2^2$$

However, we know what $v_1$ is, so we can sub that in for $v_1$ in this equation:

$$𝑣_2 = 𝛽_2 ⋅ (𝛽_2 ⋅ 𝑣_{0} + (1 − {\beta_2})⋅ 𝑔_1^2) + (1 − {\beta_2})⋅ 𝑔_2^2$$

Now we are multiplying $v_0$ by $\beta_2$ **twice** once we hit timestep 2. We can keep applying this same expansion as we advance in time.

<br>

## Bias Correction

These moments are **BIASED** early on because they are initialized at 0 (look back at the pseudocode diagram for a reminder). This makes it so that our moment estimates are biased low (the $\beta * v_0$ just goes to 0, leaving us with only $(1 - \beta) * g_t$).

Hence, we have to correct the bias:

$$\hat{𝑚_𝑡}= 𝑚_𝑡∕(1 − {\beta_1^𝑡})$$ 
$$\hat{𝑣_𝑡}= 𝑣_𝑡∕(1 − {\beta_2^𝑡})$$ 

<br>

But where does this bias correction come from?

First, we rewrite the moment as a function of all previous gradients:

$$𝑣_𝑡 = (1 − {\beta _2}) \sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} ⋅ 𝑔_𝑖^2$$

We want to know how the *expected* value of the moment ($v_t$) relates to the true gradient ($g_t^2$):

$$𝔼[𝑣_𝑡] = 𝔼[(1 − {\beta _2}) \sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} ⋅ 𝑔_𝑖^2]$$

<br>

$$𝔼[𝑣_𝑡] = 𝔼[𝑔_𝑖^2] ⋅ (1 − {\beta _2}) \sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} + \zeta$$

What the paper fails to make explicit is that the summation here is actually a **geometric series** (i.e., when summing those values, sequential terms are related to each other by a common ratio, in this case, ${\beta_2}$). Theis summation is equivalent to:

$$\sum_{𝑖=1}^𝑡 {\beta_2^{𝑡 −1}} = \frac{1 − {\beta_2^𝑡}}{1 − {\beta_2}}$$

Therefore, we can sub in $\frac{1 − {\beta_2^𝑡}}{1 − {\beta_2}}$ for the summation in our equation.

$$𝔼[𝑣_𝑡] = 𝔼[𝑔_𝑖^2] ⋅ (1 − {\beta _2}) \frac{1 − {\beta_2^𝑡}}{1 − {\beta_2}} + \zeta$$

And then we can arrive at the final equation shown in the paper by cancelling out the $1 - {\beta_2}$:

$$𝔼[𝑣_𝑡] = 𝔼[𝑔_𝑖^2] ⋅ (1 − {\beta_2^𝑡}) + \zeta$$

So now we know that the expected value of the gradient is related to the true gradient by a factor of $1 - {\beta_2^t}$. That's why we divide $m_t$ and $v_t$ by $1 - {\beta_2^t}$.

<br>

# Step Size

The change in our step size at time t is:

$${\Delta_𝑡} = \alpha ⋅ \frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}}}$$

, where 𝛼 is a “maximum” step size parameter. If you want to take N steps to the optimum that is D distance away, then $\alpha \approx \frac{𝐷}{𝑁}$

<br>

## Gradient Scaling

Since the final step equation divides the estimated mean by the estimated variance (1st moment / 2nd moment), any gradient scaling cancels out.​

$$\frac{𝑐 ⋅ \hat{𝑚_𝑡}}{\sqrt{(𝑐^2  ⋅ \hat{𝑣_𝑡}}}= \frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}}}$$

This means that, no matter what scale your inputs are, Adam will take the same step size – only $\alpha$ affects the step size.

If the Adam Optimizer is invariable to gradient scaling, does normalizing the data help convergence?

**Yes!** Normalization changes the scale of parameters in relation to each other, not the overall scale. So, normalizing the input can still improve updates.

<img width="1000" height="471" alt="image" src="https://github.com/user-attachments/assets/40d50f69-9e87-4009-8ac3-f61fb48d7f3a" /> 


The effective step sizes are approximately bound to the step size hyperparameter. They are bounded by relating the decay rates:

$$|\Delta_t| \leq \alpha  ⋅ \frac{1 - \beta_1}{\sqrt{1 - \beta_2}}, \qquad \text{if }  (1 - \beta_1) \gt \sqrt{1 - \beta_2}$$   
  
<p align="center">$$|\Delta_t| \leq \alpha, \qquad \text{if }  (1 - \beta_1) \leq \sqrt{1 - \beta_2}$$          

This keeps us from taking too large of a step and overshooting and destabilizing our optimization. 

Severe sparsity happens when a gradient has been zero at all timesteps except at the current timestep. Below is a plot which shows the "break point" between $\beta s$ that result in the first case versus $\beta s$ that result in the second case.

<img width="933" height="716" alt="image" src="https://github.com/user-attachments/assets/a13403cc-6477-419d-b911-4d720a19c1c1" />


<br>

## Step Size Annealing

$\frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}}}$ is considered to be a signal-to-noise ratio (SNR). When SNR is small, the step size decreases. SNR typically decreases when approaching an optimum, where we want smaller effective steps. This is what we showed and described when explaining the second moment.


# Updating the Parameters

Now we can finally update our parameter values! (𝜀 is there to prevent dividing by 0.)

$${\theta_t} = {\theta_{𝑡 − 1}} − {\alpha _t}  ⋅ \frac{\hat{𝑚_𝑡}}{\sqrt{\hat{𝑣_𝑡}} + \epsilon}$$

<br>

# Drawing Some Comparisons

Now that we know how the Adam Optimizer works, how does it differ from its inspirations?

## Gradient Descent

Gradient descent blindly follows the gradient without accounting for elliptical shape of the loss surface. Conversely, Adam adjusts step size to better navigate loss surface.

<img width="1136" height="871" alt="image" src="https://github.com/user-attachments/assets/99e144db-21ea-46d2-8fd4-724852c0a451" />

<br>

## AdaGrad

AdaGrad (**Ada**ptive **Grad**ient Algorithm) directly inspired the Adam Optimizer's adaptive step size. AdaGrad computes the sum of squared gradients and uses that to update parameters (without momentum). This sum can grow increasingly large over time, making the step size very small (because there is no decay of far-off gradients), delaying convergence.

$${\theta_t} = {\theta_{𝑡 − 1}} − {\alpha _t}  ⋅ \frac{{g_𝑡}}{\sqrt{\sum{g_t^2}} + \epsilon}$$

<br>

## RMSProp

RMSProp was developed to solve the problem of AdaGrad's diminishing learning rate. The key change here is the moving average. Since RMSProp give more weight to recent gradients, it adapts effectively without decreasing the learning rate too far.

$${v_t} = {\beta v_{t-1}} + (1 - \beta)g_t^2$$

$${\theta_t} = {\theta_{𝑡 − 1}} − {\alpha _t}  ⋅ \frac{{g_𝑡}}{\sqrt{v_t} + \epsilon}$$

Thought exercise: which parts of Adam come from RMSProp?

<img width="1671" height="478" alt="image" src="https://github.com/user-attachments/assets/f7b61d63-5c97-42f7-a2fd-19d5d5971419" />

<br>

## Second Order Methods

AdaGrad, RMSProp, and Adam are all first order methods. This means they use gradient information to minimize loss. Their low computational cost makes them ideal for deep learning.

Conversely, second order methods like Newton's method use curvature information through the Hessian matrix. They converge faster at a higher computational cost. The Hessian matrices calculate the gradients almost exactly, allowing for more precise steps to reach the optimum. However, they are extremely costly, with a cost of O(n3)  compared to O(n) for Adam. Newton's Method is also an approximation of the Normal equation, which is the theoretical perfect model if the loss surface is perfectly quadratic. However, this model is unrealistic (good luck finding a loss surface that's perfectly quadratic!), doesn’t work on non-linear models, and is even more costly than Newton’s.

Adam balances reaching the optimum quickly with cost (O(n)).

<br>

## Speed Comparisons

In the paper, the authors also perform several experiments to test the efficiency of Adam. Long story short: Adam performs better in every experiment.

## Logistic Regression

<img width="1370" height="750" alt="image" src="https://github.com/user-attachments/assets/6b1df982-1188-4702-b90a-6ca820e22016" />

## Multi-Layer Neural Networks

<img width="1372" height="794" alt="image" src="https://github.com/user-attachments/assets/11eace74-16f0-4519-a85b-df62f2f5802d" />

## Convolutional Neural Networks

<img width="1358" height="753" alt="image" src="https://github.com/user-attachments/assets/d0490f8f-2ffd-4d8f-a720-599db664d6e6" />
/>

## Variational Auto-Encoders

<img width="1396" height="736" alt="image" src="https://github.com/user-attachments/assets/b1e34496-8cf6-4de3-a205-7fb2a0c14709" />


In the exercise for this week, we will also be having you compare Adam to other optimizers.
