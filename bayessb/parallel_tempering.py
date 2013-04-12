from bayessb import MCMC, MCMCOpts
import collections
import numpy as np
from pylab import *
from scipy.stats import norm
from scipy.special import gamma

class PT_MCMC(MCMC):

    def __init__(self, opts, num_chains):
        max_temp = -6
        min_temp = 0
        self.swap_period = 20
        self.chains = []
        temps = np.logspace(-6, 0, num_chains)

        for i, temp in enumerate(temps):
            chain = MCMC(opts)
            chain.options.thermo_temp = temp
            self.chains.append(chain)



    def estimate(self):
        """Parallel tempering MCMC algorithm (see Geyer, 1991)."""
        while self.iter < self.options.nsteps:

            # perform update for each chain
            for chain in chains:
                generate_new_position(chain)
                check_acceptance()
                update()

            if self.iter % swap_period == 0:
                # attempt to swap two of the chains
                # pick i, j randomly from len(chains)
                # Calculate odds ratio:
                x_i = chain_i.position
                x_j = chain_j.position
                pi_xi = chain_i.posterior
                pj_xj = chain_j.posterior
                #(,,pi_xj) = chain_i.calculate_posterior(position=x_j)
                #(,,pj_xi) = chain_j.calculate_posterior(position=x_i)
                #r = (pi_xj * pj_xi)/(pi_xi * pj_xi)

                # Draw another random number, alpha
                # check if alpha < r
                # if yes, accept
                # otherwise, reject

            # choose test position and calculate posterior there
            self.test_position = self.generate_new_position()
            (self.test_posterior, self.test_prior, self.test_likelihood) = \
                self.calculate_posterior(self.test_position)

Parameter = collections.namedtuple('Parameter', 'name value')

def step(mcmc):
    """Useful step function."""
    if mcmc.iter % 200 == 0:
        print 'iter=%-5d  sigma=%-.3f  T=%-.3f ' \
              'glob_acc=%-.3f  lkl=%g  prior=%g  post=%g' % \
              (mcmc.iter, mcmc.sig_value, mcmc.T,
               mcmc.acceptance/(mcmc.iter+1.), mcmc.accept_likelihood,
               mcmc.accept_prior, mcmc.accept_posterior)

class GaussianFit():
    """Dummy model for fitting multivariate gaussians."""
    def __init__(self, means, variances):
        if len(means) != len(variances):
            raise Exception("Mismatch between means and variances")
        num_dimensions = len(means)
        self.means = means
        self.variances = variances
        self.parameters = []
        for i in range(num_dimensions):
            #self.norm = norm(loc=means[i], scale=sqrt(variances[i]))
            self.parameters.append(Parameter('p%d' % i, means[i]))

    def likelihood(self, mcmc, position):
        return np.sum(((position - self.means)**2) / (2 * self.variances))
        #return self.norm.pdf(position[0])

class Beta_Fit():
    """Dummy model for fitting a beta distribution."""
    def __init__(self, alpha, beta):
        self.parameters = [Parameter('p', 0)]
        self.a = alpha
        self.b = beta

    def likelihood(self, mcmc, position):
        #beta.pdf(x, a, b) = gamma(a+b)/(gamma(a)*gamma(b)) * x**(a-1) *
        #(1-x)**(b-1),
        const = gamma(self.a + self.b) / (gamma(self.a)*gamma(self.b))
        return const * (x ** (self.a - 1)) * ((1 - x)**(self.b - 1))

def test_gaussian_class():
    means = np.random.rand(5)
    variances = np.random.rand(5)
    g = GaussianFit(means, variances)
    # check that the model has parameters and each param has value attribute
    for p in g.parameters:
        a = p.value
    assert True

def test_gaussian_likelihood():
    num_dimensions = 5
    means = np.random.rand(num_dimensions)
    variances = np.random.rand(num_dimensions)
    g = GaussianFit(means, variances)
    print g.likelihood(None, np.ones(num_dimensions))
    assert True

def test_fit_gaussian():
    num_dimensions = 1
    # Create the dummy model
    means = 10 ** np.random.rand(num_dimensions)
    variances = np.random.rand(num_dimensions)
    g = GaussianFit(means, variances)

    # Create the options
    nsteps = 20000
    opts = MCMCOpts()
    opts.model = g
    opts.estimate_params = g.parameters
    opts.initial_values = np.ones(num_dimensions)
    opts.nsteps = nsteps
    opts.anneal_length = nsteps/10
    opts.T_init = 1
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 1
    opts.likelihood_fn = g.likelihood
    opts.step_fn = step

    # Create the PT object
    mcmc = MCMC(opts)
    mcmc.initialize()
    mcmc.estimate()
    mcmc.prune(nsteps/10, 100)

    #mixed_positions = mcmc.positions[nsteps/10:]
    #mixed_accepts = mixed_positions[mcmc.accepts[nsteps/10:]]
    ion()
    for i in range(mcmc.num_estimate):
        #mean_i = np.mean(mixed_accepts[:, i])
        #var_i = np.var(mixed_accepts[:, i])
        mean_i = np.mean(mcmc.positions[:,i])
        var_i = np.var(mcmc.positions[:, i])
        print "True mean: %f" % means[i]
        print "Sampled mean: %f" % mean_i
        print "True variance: %f" % variances[i]
        print "Sampled variance: %f" % var_i
        figure()
        (heights, points, lines) = hist(mcmc.positions[:,i], bins=50, normed=True)
        plot(points, norm.pdf(points, loc=means[i], scale=sqrt(variances[i])), 'r')

    import ipdb; ipdb.set_trace()
    mean_err = np.zeros(len(mcmc.positions))
    var_err = np.zeros(len(mcmc.positions))
    for i in range(len(mcmc.positions)):
        mean_err[i] = (means[0] - np.mean(mcmc.positions[:i+1,0]))
        var_err[i] = (variances[0] - np.var(mcmc.positions[:i+1,0]))
    figure()
    plot(mean_err)
    plot(var_err)

    import ipdb; ipdb.set_trace()
    return mcmc

def fit_beta():
    num_dimensions = 1
    # Create the dummy model
    means = 10 ** np.random.rand(num_dimensions)
    variances = np.random.rand(num_dimensions)
    g = GaussianFit(means, variances)

    # Create the options
    nsteps = 100000
    opts = MCMCOpts()
    opts.model = g
    opts.estimate_params = g.parameters
    opts.initial_values = np.ones(num_dimensions)
    opts.nsteps = nsteps
    opts.anneal_length = nsteps/10
    opts.T_init = 1
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 2
    opts.likelihood_fn = g.likelihood
    opts.step_fn = step

    # Create the PT object
    num_temps = 13
    mcmc = MCMC(opts)
    mcmc.initialize()
    mcmc.estimate()
    #pt = PT_MCMC(opts, num_temps)

    mixed_positions = mcmc.positions[nsteps/10:]
    mixed_accepts = mixed_positions[mcmc.accepts[nsteps/10:]]
    ion()
    for i in range(mcmc.num_estimate):
        mean_i = np.mean(mixed_accepts[:, i])
        var_i = np.var(mixed_accepts[:, i])
        print "True mean: %f" % means[i]
        print "Sampled mean: %f" % mean_i
        print "True variance: %f" % variances[i]
        print "Sampled variance: %f" % var_i
        figure()
        (heights, points, lines) = hist(mixed_accepts[:, i], bins=200, normed=True)
        plot(points, norm.pdf(points, loc=means[i], scale=sqrt(variances[i])), 'r')
    import ipdb; ipdb.set_trace()
    return mcmc

