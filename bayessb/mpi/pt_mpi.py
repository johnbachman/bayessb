"""
Example template for how to implement parallel tempering algorithm using MPI.

Run with, e.g.::

    mpiexec -np 5 python hello_mpi.py
"""

from mpi4py import MPI
import numpy as np
from StringIO import StringIO
from bayessb.tests import TwoDGaussianFit
from bayessb import MCMC, MCMCOpts
import math
import pickle

# Define a useful step function that can handling logging

class PT_MPI_Master(object):
    def __init__(self, comm, options, swap_period, num_chains):
        """Document me"""
        self.comm = comm
        self.options = options
        self.swap_period = swap_period
        self.num_chains = num_chains
        # Initialize log of execution
        self.log = StringIO()
        self.log.write("rank = %d --------------------------\n" % rank)
        self.random = np.random.RandomState(options.seed)

    def run(self):
        """Document me"""
        # Iterate for the total number of swap cycles
        num_swaps = self.options.nsteps / self.swap_period
        for swap_iter in range(num_swaps):
            # Pick two chains to swap
            i = self.random.randint(1, self.num_chains - 1)
            j = i+1
            # Tell everyone except i and j to proceed
            for target_id in range(1, self.num_chains):
                if target_id == i or target_id == j:
                    self.log.write("Master tells %d and %d to swap!\n" % (i, j))
                    self.comm.send(i, dest=target_id, tag=1)
                else:
                    self.log.write("Master tells %d to continue!\n" %
                            (target_id))
                    self.comm.send("continue", dest=target_id, tag=1)
        # We're done, so tell everyone to stop
        for target_id in range(1, num_chains):
            self.comm.send("stop", dest=target_id, tag=1)
        # Write the log
        with open('pt_mpi_log_%d.log' % rank, 'w') as f:
            f.write(self.log.getvalue())

class PT_MPI_Worker(object):
    def __init__(self, comm, options, swap_period):
        """Document me"""
        self.comm = comm
        self.swap_period = swap_period
        # Set the step function
        options.step_fn = self.step
        # Initialize this chain
        self.chain = MCMC(options)
        self.chain.initialize()
        self.chain.iter = self.chain.start_iter
        # Initialize log of execution
        self.log = StringIO()
        self.log.write("rank = %d --------------------------\n" % rank)

    def step(self, mcmc):
        """Useful step function."""
        if mcmc.iter % 1 == 0:
            self.log.write('iter=%-5d  sigma=%-.3f  T=%-.3f ' \
                  'glob_acc=%-.3f  lkl=%g  prior=%g  post=%g\n' % \
                  (mcmc.iter, mcmc.sig_value, mcmc.T,
                   mcmc.acceptance/(mcmc.iter+1.), mcmc.accept_likelihood,
                   mcmc.accept_prior, mcmc.accept_posterior))

    def run(self):
        """Document me"""
        while True:
            # Check if we're going to swap
            swap_cmd = self.comm.recv(source=0, tag=1)

            # Processes that should proceed without swapping this round:
            if swap_cmd == "continue":
                # Run swap_period steps
                self.log.write("Chain at rank %d (temp %g) running %d steps\n" %
                        (rank, self.chain.options.T_init, self.swap_period))
                self.chain.estimate_nsteps(self.swap_period)
            # Processes that are due to terminate:
            elif swap_cmd == "stop":
                self.log.write("Chain at rank %d (temp %g) terminating\n" %
                        (rank, self.chain.options.T_init))
                break
            # Processes that are due to swap:
            else:
                # Run swap_period-1 steps...
                self.log.write("Chain at rank %d (temp %g) running %d steps\n" %
                        (rank, self.chain.options.T_init, self.swap_period - 1))
                self.chain.estimate_nsteps(self.swap_period - 1)
                # ...then do a swap for the final step.
                # Get the lower index of the pair to be swapped, which was sent
                # in the swap self.command.
                i = int(swap_cmd)
                # First, handle the case where we are the process with the lower
                # index, and we will manage the swap
                if rank == i:
                    self.lower_swap(i)
                # Now, handle the case where we are the higher index, and will
                # participate in (but not manage) the swap
                elif rank == i+1:
                    self.upper_swap(i)
        # -- end while loop (on stop command from master) --
        # Save the chain
        with open('pt_mpi_chain_%d.mcmc' % rank, 'w') as f:
            self.chain.options.likelihood_fn = None
            self.chain.options.prior_fn = None
            self.chain.options.step_fn = None
            pickle.dump(self.chain, f)
        # Write the log
        with open('pt_mpi_log_%d.log' % rank, 'w') as f:
            f.write(self.log.getvalue())

    def lower_swap(self, i):
        """Document me"""
        self.log.write("Chain at rank %d (temp %g) swapping with %d\n" %
                (rank, self.chain.options.T_init, i+1))
        # 1. The j chain will need the i position to calculate the j
        #    test posterior, so send it over
        self.comm.send(self.chain.position, dest=i+1, tag=1)
        # 2. From the j chain, we'll need the j position
        #    to calculate the probability of the j position for
        #    this chain (pi_xj)
        self.chain.test_position = self.comm.recv(source=i+1, tag=2)
        # Calculate the test posterior, pi_xj
        (self.chain.test_posterior, self.chain.test_prior,
         self.chain.test_likelihood) = \
               self.chain.calculate_posterior(position=self.chain.test_position)
        pi_xj = self.chain.test_posterior
        pi_xi = self.chain.accept_posterior
        # 3. Now, receive the test posterior and accept posterior from
        #    chain j (pj_xi and pj_xj, respectively)
        (pj_xi, pj_xj) = self.comm.recv(source=i+1, tag=3)
        # 4. Make the decision to swap or not!
        self.chain.delta_posterior = (pi_xj + pj_xi) - (pi_xi + pj_xj)
        # Criterion for swap acceptance is similar to Metropolis
        # algorithm. If the swap is an increase in probability, just
        # do it.
        if self.chain.delta_posterior < 0:
            self.log.write("Swap accepted!\n")
            self.chain.accept_move()
            self.comm.send(True, dest=i+1, tag=4) # Tell the j chain
        # Otherwise, choose a random number to decide whether to swap:
        else:
            swap_alpha = self.chain.random.rand()
            # Accept the swap
            if math.e ** -self.chain.delta_posterior > swap_alpha:
                self.log.write("Swap accepted!\n")
                self.chain.accept_move()
                self.comm.send(True, dest=i+1, tag=4) # Tell the j chain
            # Reject the swap
            else:
                self.log.write("Swap rejected!\n")
                self.chain.reject_move()
                self.comm.send(False, dest=i+1, tag=4) # Tell the j chain
        self.chain.iter += 1

    def upper_swap(self, i):
        """Document me"""
        self.log.write("Chain at rank %d (temp %g) swapping with %d\n" %
                (rank, self.chain.options.T_init, i))
        # 1. The j chain needs the i position to calculate the j
        # test posterior (pj_xi)
        self.chain.test_position = self.comm.recv(source=i, tag=1)
        # 2. The j chain then sends its position for the i chain
        #    to calculate pi_xj
        self.comm.send(self.chain.position, dest=i, tag=2)
        # Calculate the test posterior, pj_xi
        (self.chain.test_posterior, self.chain.test_prior,
         self.chain.test_likelihood) = \
               self.chain.calculate_posterior(position=self.chain.test_position)
        # 3. Send the test and accept posterior to chain i (pj_xi and
        #    pj_xj, respectively)
        self.comm.send((self.chain.test_posterior, self.chain.accept_posterior),
                  dest=i, tag=3)
        # 4. Await the decision!
        do_swap = self.comm.recv(source=i, tag=4)
        if do_swap:
            self.log.write("Swap accepted!\n")
            self.chain.accept_move()
        else:
            self.log.write("Swap rejected!\n")
            self.chain.reject_move()
        self.chain.iter += 1

if __name__ == '__main__':
    # The total number of MCMC steps to run
    nsteps = 40000
    # Frequency for proposing swaps
    swap_period = 20
    # Temperature range
    min_temp = 1
    max_temp = 100

    # The communicator to use
    comm = MPI.COMM_WORLD
    # Number of chains/workers in the whole pool
    num_chains = comm.Get_size()
    # The rank of this chain (0 is the master, others are workers)
    rank = comm.Get_rank()

    # Set up the 2-D gaussian model
    means_x = [ 0.1, 0.5, 0.9,
                0.1, 0.5, 0.9,
                0.1, 0.5, 0.9]
    means_y = [0.1, 0.1, 0.1,
               0.5, 0.5, 0.5,
               0.9, 0.9, 0.9]
    sd = 0.01
    tdg = TwoDGaussianFit(means_x, means_y, sd ** 2)

    # Create temperature array based on number of workers (excluding master)
    temps = np.logspace(np.log10(min_temp), np.log10(max_temp), num_chains-1)

    # Create the options
    opts = MCMCOpts()
    opts.model = tdg
    opts.estimate_params = tdg.parameters
    opts.initial_values = [1.001, 1.001]
    opts.nsteps = nsteps
    opts.anneal_length = 0 # necessary so cooling does not occur
    opts.T_init = temps[rank - 1] # Use the temperature for this worker
    opts.use_hessian = False
    opts.seed = 1
    opts.norm_step_size = 0.1
    opts.likelihood_fn = tdg.likelihood

    # The master coordinates when swaps occur ---------
    if rank == 0:
        pt = PT_MPI_Master(comm, opts, swap_period, num_chains)
        pt.run()
    # Everyone else runs MCMC steps and swaps when told -----------
    else:
        pt = PT_MPI_Worker(comm, opts, swap_period)
        pt.run()


