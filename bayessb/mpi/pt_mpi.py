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

comm = MPI.COMM_WORLD
# Number of chains/workers in the whole pool
num_chains = comm.Get_size()
# The rank of this chain (0 is the master, others are workers)
rank = comm.Get_rank()
# The total number of MCMC steps to run
nsteps = 40000
# Frequency for proposing swaps
swap_period = 20

# Log of execution
log = StringIO()
log.write("rank = %d --------------------------\n" % rank)

random = np.random.RandomState(1)

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
min_temp = 1
max_temp = 100
temps = np.logspace(np.log10(min_temp), np.log10(max_temp), num_chains-1)

# Define a useful step function that can handling logging
def step(mcmc):
    """Useful step function."""
    if mcmc.iter % 1 == 0:
        log.write('iter=%-5d  sigma=%-.3f  T=%-.3f ' \
              'glob_acc=%-.3f  lkl=%g  prior=%g  post=%g\n' % \
              (mcmc.iter, mcmc.sig_value, mcmc.T,
               mcmc.acceptance/(mcmc.iter+1.), mcmc.accept_likelihood,
               mcmc.accept_prior, mcmc.accept_posterior))

# The master coordinates when swaps occur ---------
if rank == 0:
    # Iterate for the total number of swap cycles
    num_swaps = nsteps / swap_period
    for swap_iter in range(num_swaps):
        #print "%d steps completed" % (swap_iter * swap_period)
        # Pick two chains to swap
        i = random.randint(1, num_chains - 1)
        j = i+1
        # Tell everyone except i and j to proceed
        for target_id in range(1, num_chains):
            #log.write("Master tells %d to continue!\n" % (target_id))
            #comm.send("continue", dest=target_id, tag=1)
            #continue
            if target_id == i or target_id == j:
                log.write("Master tells %d and %d to swap!\n" % (i, j))
                comm.send(i, dest=target_id, tag=1)
            else:
                log.write("Master tells %d to continue!\n" % (target_id))
                comm.send("continue", dest=target_id, tag=1)
    # We're done, so tell everyone to stop
    for target_id in range(1, num_chains):
        comm.send("stop", dest=target_id, tag=1)

# Everyone else runs MCMC steps and swaps when told -----------
else:
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
    opts.step_fn = step

    # Initialize this chain with the proper temperature
    chain = MCMC(opts)
    chain.initialize()
    chain.iter = chain.start_iter

    while True:
        # Check if we're going to swap
        swap_cmd = comm.recv(source=0, tag=1)

        # Processes that should proceed without swapping this round:
        if swap_cmd == "continue":
            # Run swap_period steps
            log.write("Chain at rank %d (temp %g) running %d steps\n" %
                    (rank, chain.options.T_init, swap_period))
            chain.estimate_nsteps(swap_period)
        # Processes that are due to terminate:
        elif swap_cmd == "stop":
            log.write("Chain at rank %d (temp %g) terminating\n" %
                    (rank, chain.options.T_init))
            break
        # Processes that are due to swap:
        else:
            # Run swap_period-1 steps...
            log.write("Chain at rank %d (temp %g) running %d steps\n" %
                    (rank, chain.options.T_init, swap_period - 1))
            chain.estimate_nsteps(swap_period - 1)
            # ...then do a swap for the final step.
            # Get the lower index of the pair to be swapped, which was sent
            # in the swap command.
            i = int(swap_cmd)
            # First, handle the case where we are the process with the lower
            # index, and we will manage the swap
            if rank == i:
                log.write("Chain at rank %d (temp %g) swapping with %d\n" %
                        (rank, chain.options.T_init, i+1))
                # 1. The j chain will need the i position to calculate the j
                #    test posterior, so send it over
                comm.send(chain.position, dest=i+1, tag=1)
                # 2. From the j chain, we'll need the j position
                #    to calculate the probability of the j position for
                #    this chain (pi_xj)
                chain.test_position = comm.recv(source=i+1, tag=2)
                # Calculate the test posterior, pi_xj
                (chain.test_posterior, chain.test_prior,
                 chain.test_likelihood) = \
                         chain.calculate_posterior(position=chain.test_position)
                pi_xj = chain.test_posterior
                pi_xi = chain.accept_posterior
                # 3. Now, receive the test posterior and accept posterior from
                #    chain j (pj_xi and pj_xj, respectively)
                (pj_xi, pj_xj) = comm.recv(source=i+1, tag=3)
                # 4. Make the decision to swap or not!
                chain.delta_posterior = (pi_xj + pj_xi) - (pi_xi + pj_xj)
                # Criterion for swap acceptance is similar to Metropolis
                # algorithm. If the swap is an increase in probability, just
                # do it.
                if chain.delta_posterior < 0:
                    log.write("Swap accepted!\n")
                    chain.accept_move()
                    comm.send(True, dest=i+1, tag=4) # Tell the j chain
                # Otherwise, choose a random number to decide whether to swap:
                else:
                    swap_alpha = chain.random.rand()
                    # Accept the swap
                    if math.e ** -chain.delta_posterior > swap_alpha:
                        log.write("Swap accepted!\n")
                        chain.accept_move()
                        comm.send(True, dest=i+1, tag=4) # Tell the j chain
                    # Reject the swap
                    else:
                        log.write("Swap rejected!\n")
                        chain.reject_move()
                        comm.send(False, dest=i+1, tag=4) # Tell the j chain
                chain.iter += 1
            # Now, handle the case where we are the higher index, and will
            # participate in (but not manage) the swap
            elif rank == i+1:
                log.write("Chain at rank %d (temp %g) swapping with %d\n" %
                        (rank, chain.options.T_init, i))
                # 1. The j chain needs the i position to calculate the j
                #    test posterior (pj_xi)
                chain.test_position = comm.recv(source=i, tag=1)
                # 2. The j chain then sends its position for the i chain
                #    to calculate pi_xj
                comm.send(chain.position, dest=i, tag=2)
                # Calculate the test posterior, pj_xi
                (chain.test_posterior, chain.test_prior,
                 chain.test_likelihood) = \
                         chain.calculate_posterior(position=chain.test_position)
                # 3. Send the test and accept posterior to chain i (pj_xi and
                #    pj_xj, respectively)
                comm.send((chain.test_posterior, chain.accept_posterior),
                          dest=i, tag=3)
                # 4. Await the decision!
                do_swap = comm.recv(source=i, tag=4)
                if do_swap:
                    log.write("Swap accepted!\n")
                    chain.accept_move()
                else:
                    log.write("Swap rejected!\n")
                    chain.reject_move()
                chain.iter += 1
            else:
                raise Exception("Invalid swap command received.")
    # Save the chain
    with open('pt_mpi_chain_%d.mcmc' % rank, 'w') as f:
        chain.options.likelihood_fn = None
        chain.options.prior_fn = None
        chain.options.step_fn = None
        pickle.dump(chain, f)

# Write the log
with open('pt_mpi_log_%d.log' % rank, 'w') as f:
    f.write(log.getvalue())

