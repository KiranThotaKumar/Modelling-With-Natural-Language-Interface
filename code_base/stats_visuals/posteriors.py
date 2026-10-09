import numpy as np


def flatten_posterior(samples, log_prob, burnin_frac=0.2):

    nsteps, nwalkers, ndim = samples.shape
    log_prob_2D = log_prob.reshape(nsteps, nwalkers)
    burnin = int(burnin_frac * nsteps)
    post_samples_3D = samples[burnin:, :, :]
    post_logprob_2D = log_prob_2D[burnin:, :]
    samples_flat = post_samples_3D.reshape(-1, ndim)
    log_prob_flat = post_logprob_2D.reshape(-1)

    return samples_flat, log_prob_flat

def map_estimate(samples_flat, log_prob_flat):
    map_idx = np.argmax(log_prob_flat)
    theta_map = samples_flat[map_idx]

    return theta_map

def posterior_mean(samples_flat):
    theta_mean = np.mean(samples_flat, axis=0)

    return theta_mean

def posterior_predictive(
    samples_flat,
    forward_fn,
    nsamples,
    rng=None
):
    ...
