import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

mean_npy = np.zeros(2)
target_mean = np.array([5, 5])

mu_init = np.zeros(2)
popsize=100
xsamp = np.random.randn(popsize, 2) + mu_init[None]
target_point = np.array([5., 5.])
plt.clf()
plt.plot(xsamp[:, 0], xsamp[:, 1], 'r.', label='samp')
plt.plot(target_point[0], target_point[1], 'g.', ms=20, label='target_samp')
el = Ellipse(xy=mu_init, width=2*0.5, height=2*0.5, angle=0)
el.set_edgecolor('red')
plt.arrow(2, 2, 1, 1, head_width=.2, head_length=.2, fc='k', ec='k')
plt.gca().add_artist(el)
plt.show()

# Define the updates here

##### PD (Pathwise Derivative) Update #####

def pd_update(*, mu, target_mu, step_size=0.1, n_samples=1):
  """
    sample drawn as mu + eps * sigma
    reward =  -(sample - target_mu)**2
  """
  samples = np.random.randn(n_samples, 2) + mu[None]
  grad_estim = -np.mean(samples - target_mu[None], axis=0)
  mu = mu + step_size * grad_estim
  return mu

#### SF (Score Function) Update ####

def sf_update(*, mu, target_mu, step_size=0.1, n_samples=1):
  """
    reward = -(mu - target_mu)**2
    grad_log_mu = (sample - mu) for gaussian with std=1 (check yourself)
  """
  samples = np.random.randn(n_samples, 2) + mu[None]
  rewards = -np.sum((samples- target_mu[None])**2, axis=1)
  grad_log_mu = (samples - mu[None])
  grad_estim = np.mean(rewards[:, None] * grad_log_mu, axis=0)
  #grad_estim = (1./ n_samples) * np.dot(rewards, grad_log_mu)
  mu = mu + step_size * grad_estim
  return mu

def plot_training(update_method):
  # Visualize SF
  step_size = 0.05
  mu = np.zeros(2)
  #mu = np.random.uniform(size=(2,))
  n_samples=1

  for _ in range(20):
    plt.clf()
    plt.plot(target_point[0], target_point[1], 'g.', ms=20, label='target_samp')
    el = Ellipse(xy=mu, width=2*0.5, height=2*0.5, angle=0)
    el.set_edgecolor('red')
    plt.gca().add_artist(el)
    plt.plot(mu, 'rx', label='mean')
    plt.axis([-10, 10, -10, 10])

    plt.show()
    # plt.pause(1)
    mu = update_method(mu=mu, target_mu=target_mean, n_samples=n_samples, step_size=step_size)

