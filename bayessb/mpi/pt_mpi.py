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

comm = MPI.COMM_WORLD
# Number of chains/workers in the whole pool
num_chains = comm.Get_size()
# The rank of this chain (0 is the master, others are workers)
rank = comm.Get_rank()
# The total number of MCMC steps to run
nsteps = 1000
# Frequency for proposing swaps
swap_period = 20

# Log of execution
log = StringIO()
log.write("rank = %d --------------------------\n" % rank)

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

# The master coordinates when swaps occur ---------
if rank == 0:
    # Iterate for the total number of swap cycles
    num_swaps = nsteps / swap_period
    for swap_iter in range(num_swaps):
        # Pick two chains to swap
        i = np.random.randint(1, num_chains - 1)
        j = i+1
        # Tell everyone except i and j to proceed
        for target_id in range(1, num_chains):
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
    #opts.step_fn = step

    # Initialize this chain with the proper temperature
    chain = MCMC(opts)
    chain.initialize()

    while True:
        # Check if we're going to swap
        data = comm.recv(source=0, tag=1)

        if data == "continue":
            # Run swap_period steps
            log.write("Chain at rank %d (temp %g) running %d steps\n" %
                    (rank, chain.options.T_init, swap_period))
        elif data == "stop":
            log.write("Chain at rank %d (temp %g) terminating\n" %
                    (rank, chain.options.T_init))
            break
        else:
            # Run swap_period-1 steps
            log.write("Chain at rank %d (temp %g) running %d steps\n" %
                    (rank, chain.options.T_init, swap_period - 1))
            # Now swap for the final step
            # Get the lower index of the pair to be swapped.
            i = int(data)
            # First, handle the case where we are the lower index, and will
            # manage the swap
            if rank == i:
                log.write("Chain at rank %d (temp %g) swapping with %d\n" %
                        (rank, chain.options.T_init, i+1))
                i_pos = 1
                # 1. The j chain will need the i position to calculate the j test
                #    posterior
                comm.send(i_pos, dest=i+1, tag=1)
                # 2. From the j chain, we'll need: the j position, the j
                #    accept_posterior, and the j test_posterior
                # FIXME for now, receive j_iter
                j_pos = comm.recv(source=i+1, tag=2)
                # Do the swap
                #iter = j_pos
                #log.write("Rank %d swapped from pos %d to %d\n" %
                #          (rank, i_pos, j_pos))
            # Now, handle the case where we are the higher index, and will
            # participate in (but not manage) the swap
            elif rank == i+1:
                log.write("Chain at rank %d (temp %g) swapping with %d\n" %
                        (rank, chain.options.T_init, i))
                j_pos = 1
                # 1. The j chain will receive i position to calculate the j
                #    test posterior
                i_pos = comm.recv(source=i, tag=1)
                # 2. The j chain sends the test_posterior
                # FIXME for now, send the iter
                comm.send(j_pos, dest=i, tag=2)
                # Do the swap
                #iter = i_pos
                #log.write("Rank %d swapped from pos %d to %d\n" %
                #          (rank, j_pos, i_pos))
            else:
                raise Exception("Invalid data message received.")

with open('pt_mpi_log_%d.log' % rank, 'w') as f:
    f.write(log.getvalue())
