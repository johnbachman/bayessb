from bayessb import MCMC, MCMCOpts
import numpy as np
import math

class PT_MCMC(object):
    """Implementation of parallel tempering algorithm.

    See Geyer, "Maximum likelihood Markov Chain Monte Carlo", 1991.

    In this algorithm, a series of chains are run at different temperatures in
    parallel. They execute normally (following the Metropolis algorithm) for n
    steps (defined by the parameter ``swap_period``) and then a swap is
    proposed between two chains that neighbor each other in temperature.  If it
    is accepted, each chain adopts the position of the other. Since the swap is
    symmetric, detailed balance and ergodicity is maintained (when one
    considers that the parameter space is now the Cartesian product of the
    parameter spaces for all chains).

    In this implementation, a :py:class:`PT_MCMC` object is used to contain all
    of the chains (as instances of :py:class:`bayessb.MCMC`) and manage their
    execution, performing swaps when appropriate.

    Temperatures are given as minimum and maximum values; the intermediate
    values are interpolated on a log scale. So if ``min_temp`` is 1, and
    ``max_temp`` is 100, with three temperatures, the temperatures used are 1,
    10, 100.

    Parameters
    ----------
    opts: bayessb.MCMCOpts
        Used to initialize all of the :py:class:`bayessb.MCMC` chains in the
        temperature series.
    num_chain : int
        The number of chains/temperatures to run. Too many temperatures will
        make swapping inefficient; too few temperatures will make swaps
        unlikely to be accepted.
    max_temp : number
        The highest temperature in the series.
    min_temp : number
        The lowest temperature in the series. Should usually be 1 (the default).
    swap_period : int
        Number of steps of "regular" (Metropolis) MCMC to perform for each
        chain before proposing a swap.
    """

    def __init__(self, opts, num_chains, max_temp, min_temp=1, swap_period=20):
        self.options = opts
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.swap_period = swap_period

        self.chains = []
        self.iter = 0
        # Calculate the temperature series
        temps = np.logspace(np.log10(min_temp), np.log10(max_temp),
                                           num_chains)
        # Initialize each of the chains
        for i, temp in enumerate(temps):
            chain = MCMC(opts)
            chain.options.T_init = temp
            chain.initialize()
            chain.iter = chain.start_iter
            self.chains.append(chain)

        # Initialize arrays for storing swap info
        num_swaps = self.options.nsteps / swap_period
        self.swap_proposals = np.zeros((num_swaps, 2))
        self.x_i = np.zeros((num_swaps, len(self.options.estimate_params)))
        self.x_j = np.zeros((num_swaps, len(self.options.estimate_params)))
        self.pi_xi = np.zeros(num_swaps)
        self.pi_xj = np.zeros(num_swaps)
        self.pj_xi = np.zeros(num_swaps)
        self.pj_xj = np.zeros(num_swaps)
        self.r = np.zeros(num_swaps)
        self.swap_alphas = np.zeros(num_swaps)
        self.swap_accepts = np.zeros(num_swaps, dtype=bool)
        self.swap_rejects = np.zeros(num_swaps, dtype=bool)
        self.swap_iter = 0

    def estimate(self):
        """Parallel tempering MCMC algorithm (see Geyer, 1991)."""

        while self.iter < self.options.nsteps:
            # Perform Metropolis step for each chain
            for chain in self.chains:
                # Get a new position
                chain.test_position = chain.generate_new_position()
                # Choose test position and calculate posterior there
                (chain.test_posterior, chain.test_prior, chain.test_likelihood)\
                        = chain.calculate_posterior(chain.test_position)

                # -- METROPOLIS ALGORITHM --
                # Decide whether to accept the step
                delta_posterior = chain.test_posterior - chain.accept_posterior
                if delta_posterior < 0:
                    chain.accept_move()
                else:
                    alpha = chain.random.rand()
                    chain.alphas[chain.iter] = alpha;  # log the alpha value
                    if math.e ** (-delta_posterior/chain.T) > alpha:
                        chain.accept_move()
                    else:
                        chain.reject_move()

                # Log some interesting variables
                chain.positions[chain.iter,:] = chain.test_position
                chain.priors[chain.iter] = chain.test_prior
                chain.likelihoods[chain.iter] = chain.test_likelihood
                chain.posteriors[chain.iter] = chain.test_posterior
                chain.delta_posteriors[chain.iter] = delta_posterior
                chain.sigmas[chain.iter] = chain.sig_value
                chain.ts[chain.iter] = chain.T

                chain.iter += 1

            # Call user-callback step function on the first chain in the series
            # to track execution
            if self.chains[0].options.step_fn:
                self.chains[0].options.step_fn(self.chains[0])

            # Check if it's time to propose a swap
            if self.iter >= self.swap_period and \
               self.iter % self.swap_period == 0:
                self.propose_swap()

            self.iter += 1

    def propose_swap(self):
        """Performs the temperature-swapping step of the PT algorithm."""
        # Idea is to pick two neighboring chains to swap, so we
        # randomly pick the one with the lower index
        i = np.random.randint(len(self.chains)-1)
        j = i+1
        # Calculate odds ratio:
        chain_i = self.chains[i]
        chain_j = self.chains[j]
        x_i = chain_i.position
        x_j = chain_j.position
        pi_xi = chain_i.accept_posterior
        pj_xj = chain_j.accept_posterior
        (pi_xj, a, b) = chain_i.calculate_posterior(position=x_j)
        (pj_xi, a, b) = chain_j.calculate_posterior(position=x_i)
        # Since calculations are in logspace, the ratio becomes a difference
        r = (pi_xj + pj_xi) - (pi_xi + pj_xi)

        # Decide whether to accept the swap: similar to Metropolis
        if r < 0:
            self.accept_swap(i, j, x_i, x_j)
        else:
            swap_alpha = self.chains[0].random.rand()
            self.swap_alphas[self.swap_iter] = swap_alpha

            if math.e ** (-r) > swap_alpha:
                self.accept_swap(i, j, x_i, x_j)
            else:
                self.swap_rejects[self.swap_iter] = 1

        # Log interesting variables
        self.swap_proposals[self.swap_iter,:] = (i, j)
        self.x_i[self.swap_iter] = x_i
        self.x_j[self.swap_iter] = x_j
        self.pi_xi[self.swap_iter] = pi_xi
        self.pi_xj[self.swap_iter] = pi_xj
        self.pj_xi[self.swap_iter] = pj_xi
        self.pj_xj[self.swap_iter] = pj_xj
        self.r[self.swap_iter] = r

        # Increment the swap count
        self.swap_iter += 1

    def accept_swap(self, i, j, x_i, x_j):
        self.chains[i].position = x_j
        self.chains[j].position = x_i
        self.swap_accepts[self.swap_iter] = 1


