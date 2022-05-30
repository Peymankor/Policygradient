
# Actor-Critic and Variance Rediuction

- Q-Value function approximation
- $$\Delta w=\beta.(R_{t+1}+\gamma.Q(S_{t+1},A_{t+1};w)-Q(S_t,A_t;w_t)).\nabla_w Q(S_t,A_t;w)$$
- The policy mean parameter $\theta$ are updated after each atomic experince
- $$\Delta \theta = \alpha.\gamma^t.(\nabla{\theta} \log \pi(S_t;A_t;\theta)).Q(S_t,A_t;w))$$

- Actor-Critic with Baseline:
  $$\Delta \theta = \alpha.\gamma^t.(\nabla_{\theta} \log \pi(S_t;A_t;\theta)).(Q(S_t,A_t;w)-B(S_t))$$


## Loss function in PG

$$L^{PG}(\theta)=\mathbb{E}_t[\log \pi_{\theta}(a_t|s_t)\hat{A}_t]$$

## PPO

- "Current policy"
- "baseline policy"

### Advantage function

- How good an action is compared to the average action for a specific state.

$$A(s,a) = Q(s,a) -V(s)$$

### PPO

- To estimate a "trust region" where we can safely take reasonable steps in the right direction.

- The current policy that we want:
  $$\pi_{\theta}(a_t|s_t)$$

- The previous policy
  $$\pi_{\theta_k}(a_t|s_t)$$

- ratio
  $$r_t(\theta)= \pi_{\theta}(a_t|s_t)/\pi_{\theta_k}(a_t|s_t)$$

### PPO objective

$$L^{CLIP}(\theta)= \hat{\mathbb{E}}[\min(r_t(\theta)\hat{A}, clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)] $$



### Approximate Advantage Function

