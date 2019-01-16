import os
import copy
import time
import pathlib
import logging
import argparse
import parmed as pmd

import mdtraj as md
import numpy as np

from simtk import openmm, unit
from simtk.openmm import app
from mdtraj.reporters import NetCDFReporter
import yank
from simtk.openmm import XmlSerializer
from openmmtools import states, mcmc

class MyComposableState(GlobalParameterState):
    lambda_restraints = GlobalParameterState.GlobalParameter('lambda_restraints', standard_value=0.0)
    K_parallel = GlobalParameterState.GlobalParameter('K_parallel',
                                                    standard_value=1250*unit.kilojoules_per_mole/unit.nanometer**2)
    Kmax = GlobalParameterState.GlobalParameter('Kmax',
                                                standard_value=500*unit.kilojoules_per_mole/unit.nanometer**2)
    Kmin = GlobalParameterState.GlobalParameter('Kmin',
                                                standard_value=500*unit.kilojoules_per_mole/unit.nanometer**2)

def write_cv(context):

    for replica_index in range(simulation.n_replicas):
        state_index = simulation._replica_thermodynamic_states[replica_index]
        ss = simulation.sampler_states[replica_index]
        ss.apply_to_context(context, ignore_velocities=True)
        ss.update_from_context(context, ignore_positions=True, ignore_velocities=True)
        logger.debug(ss.collective_variables)

def main():
    parser = argparse.ArgumentParser(description='Compute a potential of mean force (PMF) for porin permeation.')
    parser.add_argument('--index', dest='index', action='store', type=int,
                    help='Index of ')

    args = parser.parse_args()
    index = args.index

    logger = logging.getLogger(__name__)
    logging.root.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    yank.utils.config_root_logger(verbose=True, log_file_path=None)
    platform = openmm.Platform.getPlatformByName('CUDA')
    integrator = openmm.LangevinIntegrator(310*unit.kelvin, 1.0/unit.picoseconds, 2*unit.femtosecond)

    # Topology
    pdbx = app.PDBxFile('mem_prot_md_system.pdbx')

    # This system contains the CVforce with parameters different than zero

    with open('openmm_system.xml', 'r') as infile:
        openmm_system = XmlSerializer.deserialize(infile.read())

    ####### Indexes of configurations in trajectory ############################
    configs = [39, 141, 276, 406, 562, 668, 833, 1109, 1272, 1417, 1456, 1471, 1537, 1645, 1777, 1882]

    ####### Indexes of states for series of replica exchange simulations #######
    limits = [( 0, 9), (10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99), (100, 109), (110, 119), (120, 129), (130, 139), (140, 149), (150, 159)]

    ####### Reading positions from mdtraj trajectory ###########################
    topology = md.Topology.from_openmm(pdbx.topology)
    t = md.load('../../steered_md/comp7/forward/seed_0/steered_forward.nc',top=topology)
    positions = t.openmm_positions(configs[index])

    thermodynamic_state_deserialized = states.ThermodynamicState(system=openmm_system,
                                                                temperature=310*unit.kelvin,
                                                                pressure=1.0 * unit.atmospheres)

    sampler_state = states.SamplerState(positions=positions, box_vectors=t.unitcell_vectors[configs[index],:,:])

    move = mcmc.LangevinDynamicsMove(timestep=2*unit.femtosecond, collision_rate= 1.0/unit.picoseconds, n_steps=2000, reassign_velocities=False)
    simulation = ReplicaExchangeSampler(mcmc_moves=move, number_of_iterations=1)
    analysis_particle_indices = topology.select('(protein and mass > 3.0) or (resname MER and mass > 3.0)')
    reporter = MultiStateReporter(checkpoint_interval=checkpoint_interval, analysis_particle_indices=analysis_particle_indices)


    # Initialize compound thermodynamic states
    protocol = {'lambda_restraints': [ i/159 for i in range(first, last+1)],
                'K_parallel': [500*unit.kilojoules_per_mole/unit.nanometer**2 for i in range(first, last+1)],
                'Kmax': [500*unit.kilojoules_per_mole/unit.nanometer**2 for i in range(first, last+1)],
                'Kmin': [500*unit.kilojoules_per_mole/unit.nanometer**2 for i in range(first, last+1)]}

    my_composable_state = MyComposableState()
    compound_states = states.create_thermodynamic_state_protocol(thermodynamic_state_deserialized,
                                                                protocol=protocol,
                                                                composable_states=[my_composable_state])

    simulation.create(thermodynamic_states=compound_states,
                      sampler_states=sampler_state,
                      storage=reporter)

    simulation.equilibrate(100000, mcmc_moves=move)
    # Run the simulation
    simulation.run()
    ts = simulation._thermodynamic_states[0]
    context, _ = openmmtools.cache.global_context_cache.get_context(ts)
    i/159 for i in range(first, last+1)
    files_names = ['state_' + i + '.log' for i in range(simulation.n_replicas)]
    files = []
    for i, file in enumerate(files_names):
        files.append(open(file, 'w'))

    mpi.distribute(write_cv(context), range(simulation.n_replicas), send_results_to=None)

    for i in range(10000):
       simulation.extend(n_iterations=1)
       mpi.distribute(write_cv(context), range(simulation.n_replicas), send_results_to=None)

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    main()
