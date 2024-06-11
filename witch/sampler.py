import corner
import emcee
import numpy as np

from . import forward_modeling as fm


def parhelper(theta, fixed_pars, fix_pars_idx):
    _theta = np.empty(len(theta) + len(fixed_pars))

    _theta[fix_pars_idx] = fixed_pars
    _theta[~fix_pars_idx] = theta

    return _theta


def sample(
    model, todvec, skymap, nwalk=100, nstep=1000, nburn=500, usepval=True, **kwargs
):
    params = np.array(model.pars)
    priors = model.priors

    fixed_pars_idx = ~np.array(model.to_fit)
    fixed_pars_val = params[fixed_pars_idx]

    ndims = np.count_nonzero(model.to_fit)
    tods = fm.make_tod_stuff(todvec, skymap)

    jitlike = fm.jax.jit(lambda x: fm.loglike(model, x, tods))
    jitlike(np.random.rand(ndims))

    def loglike(theta):
        _theta = parhelper(theta, fixed_pars_val, fixed_pars_idx)
        return jitlike(_theta)

    def logprior(theta):
        prob = np.array([priors[pi].logpdf(theta[pi]) for pi in range(ndims)])
        return np.sum(prob)

    def logpost(theta):
        lp = logprior(theta)
        if np.isfinite(lp):
            return lp + loglike(theta)
        return -np.inf

    sampler = emcee.EnsembleSampler(nwalk, ndims, logpost)

    print(f"* Running burn-in stage [{nburn} steps]")

    if usepval:
        state = np.empty((nwalk, ndims))
        for pi, pp in enumerate(params[model.to_fit]):
            ss = 0.01 * np.random.rand(nwalk)
            state[:, pi] = pp + ss if pp == 0.00 else pp * (1.00 + ss)
    else:
        state = np.array([pp.rvs(size=nwalk) for pp in priors]).T
    state = sampler.run_mcmc(state, nburn, progress=kwargs.get("progress", True))

    print(f"* Running sampling stage [{nstep} steps]")
    sampler.reset()
    sampler.run_mcmc(state, nstep, progress=kwargs.get("progress", True))

    model.cur_round += 1
    return sampler


def update_noise(
    model, todvec, skymap, samples, noise_class, *noise_args, **noise_kwargs
):
    tods = fm.make_tod_stuff(todvec, skymap, x0=model.x0, y0=model.y0)

    pars = np.array(
        [corner.quantile(samples[:, p], 0.50)[0] for p in range(samples.shape[1])]
    )
    pred = fm.model(model.xyz, *model.n_struct, model.dz, model.beam, *pars)

    for ti, tod in enumerate(tods):

        x = tod[0].copy()
        y = tod[1].copy()
        pred_tod = fm.bilinear_interp(
            x, y, model.xyz[0].ravel(), model.xyz[1].ravel(), pred
        )

        todvec.tods[ti].set_noise(
            noise_class,
            todvec.tods[ti].info["dat_calib"] - pred_tod,
            *noise_args,
            **noise_kwargs,
        )
        del x, y, pred_tod
