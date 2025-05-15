"""
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Shape Optimization
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module computes the dynamics of a capsule re-entering the atmosphere of the Earth, using a variety of integrator
and propagator settings.  For each run, the differences w.r.t. a benchmark propagation are computed, providing a proxy
for setting quality. The benchmark settings are currently defined semi-randomly, and are to be analyzed/modified.

The trajectory of the capsule is heavily dependent on the shape and orientation of the vehicle. Here, the shape is
determined here by the five parameters, which are used to compute the aerodynamic accelerations on the vehicle using a
modified Newtonian flow (see Dirkx and Mooij, "Conceptual Shape Optimization of Entry Vehicles" 2018). The bank angle
and sideslip angles are set to zero. The vehicle shape and angle of attack are defined by values in the vector shape_parameters.

The vehicle starts 120 km above the surface of the planet, with a speed of 7.83 km/s in an Earth-fixed frame (see
getInitialState function).

The propagation is terminated as soon as one of the following conditions is met (see 
get_propagation_termination_settings() function):
- Altitude < 25 km
- Propagation time > 24 hr

This propagation assumes only point mass gravity by the Earth and aerodynamic accelerations.

The entries of the vector 'shape_parameters' contains the following:
- Entry 0:  Nose radius
- Entry 1:  Middle radius
- Entry 2:  Rear length
- Entry 3:  Rear angle
- Entry 4:  Side radius
- Entry 5:  Constant Angle of Attack

Details on the outputs written by this file can be found:
- benchmark data: comments for 'generateBenchmarks' function
- results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"
- files defining the points and surface normals of the mesg used for the aerodynamic analysis (save_vehicle_mesh_to_file)

Frequent warnings and/or errors that might pop up:
* One frequent warning could be the following (mock values):
    "Warning in interpolator, requesting data point outside of boundaries, requested data at 7008 but limit values are
    0 and 7002, applying extrapolation instead."
It can happen that the benchmark ends earlier than the regular simulation, due to the smaller step size. Therefore,
the code will be forced to extrapolate the benchmark states (or dependent variables) to compare them to the
simulation output, producing a warning. This warning can be deactivated by forcing the interpolator to use the boundary
value instead of extrapolating (extrapolation is the default behavior). This can be done by setting:

    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

* One frequent error could be the following:
    "Error, propagation terminated at t=4454.723896, returning propagation data up to current time."
    This means that an error occurred with the given settings. Typically, this implies that the integrator/propagator
    combination is not feasible. It is part of the assignment to figure out why this happens.

* One frequent error can be one of:
    "Error in RKF integrator, step size is NaN"
    "Error in ABM integrator, step size is NaN"
    "Error in BS integrator, step size is NaN"

This means that a variable time-step integrator wanting to take a NaN time step. In such cases, the selected
integrator settings are unsuitable for the problem you are considering.

NOTE: When any of the above errors occur, the propagation results up to the point of the crash can still be extracted
as normal. It can be checked whether any issues have occured by using the function

dynamics_simulator.integration_completed_successfully

which returns a boolean (false if any issues have occured)

* A frequent issue can be that a simulation with certain settings runs for too long (for instance if the time steo
becomes excessively small). To prevent this, you can add an additional termination setting (on top of the existing ones!)

    cpu_tim_termination_settings = propagation_setup.propagator.cpu_time_termination(
        maximum_cpu_time )

where maximum_cpu_time is a varaiable (float) denoting the maximum time in seconds that your simulation is allowed to
run. If the simulation runs longer, it will terminate, and return the propagation results up to that point.

* Finally, if the following error occurs, you can NOT extract the results up to the point of the crash. Instead,
the program will immediately terminate

    SPICE(DAFNEGADDR) --

    Negative value for BEGIN address: -214731446

This means that a state is extracted from Spice at a time equal to NaN. Typically, this is indicative of a
variable time-step integrator wanting to take a NaN time step, and the issue not being caught by Tudat.
In such cases, the selected integrator settings are unsuitable for the problem you are considering.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################


# import sys
# sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice as spice_interface
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import environment
from tudatpy import numerical_simulation
from tudatpy.math import interpolators
import tudatpy.util as util

# Problem-specific imports
import CapsuleEntryUtilities as Util
from CapsuleEntryProblem import ShapeOptimizationProblem

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
shape_parameters = [8.148730872315355,
                    2.720324489288032,
                    0.2270385167794302,
                    -0.4037530896422072,
                    0.2781438040896319,
                    0.4559143679738996]
# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Set termination conditions
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 25.0E3  # m
# Set vehicle properties
capsule_density = 250.0  # kg m-3

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Earth']
# Define coordinate system
global_frame_origin = 'Earth'
global_frame_orientation = 'J2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create and add capsule to body system
Util.add_capsule_to_body_system(bodies,
                                shape_parameters,
                                capsule_density)

#######################################################################
### TERMINATION SETTINGS AND RETRIEVING DEPENDENT VARIABLES TO SAVE ###
#######################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude)

################################
### DESIGN SPACE EXPLORATION ###
################################

# List of minimum and maximum values for each design parameter (shape parameter)
decision_variable_range = \
    [[ 3.5, 2.0, 0.1, np.deg2rad(-55.0), 0.01, 0.0 ],
     [ 10.0, 3.0, 5.0, np.deg2rad(-10.0), 0.5, np.deg2rad(30.0) ]]

# NOTE TO STUDENTS: HERE YOU INPUT WHAT DESIGN SPACE EXPLORATION METHOD YOU USE
design_space_method = 'monte_carlo'

number_of_parameters = len(decision_variable_range[0])

# The number of Monte Carlo simulations is defined, as well as the seed which
# is passed to the MT19937 BitGenerator
if design_space_method == 'monte_carlo_one_at_a_time':
    number_of_simulations_per_parameter = 50
    number_of_simulations = number_of_parameters *  number_of_simulations_per_parameter
    nominal_parameters = [ 8.0, 1.5, 3.0, np.deg2rad(-30.0), 0.1, np.deg2rad(15.0) ]
    random_seed = 42 # ;)
    np.random.seed(random_seed) # Slightly outdated way of doing this, but works
    print('\n Random Seed :', random_seed, '\n')

elif design_space_method == 'monte_carlo':
    number_of_simulations = 100
    random_seed = 42 # ;)
    np.random.seed(random_seed) # Slightly outdated way of doing this, but works
    print('\n Random Seed :', random_seed, '\n')

# elif design_space_method == 'factorial_design':
#     # no_of_factors equals the number of parameters, all interactions are
#     # included somewhere in the factorial design
#     no_of_factors = number_of_parameters
#     no_of_levels = 2
#     # Function that creates the yates_array
#     yates_array = util.get_yates_array(no_of_factors,no_of_levels)
#     number_of_simulations = len(yates_array)
#
#     # Evenly distributed set of values between minimum and maximum value
#     # defined earlier
#     design_variable_arr = np.zeros((no_of_levels, no_of_factors))
#     for par in range(no_of_factors):
#         design_variable_arr[:, par] = np.linspace(decision_variable_range[0][par], decision_variable_range[1][par], no_of_levels, endpoint=True)

parameters = dict()
objectives_and_constraints = dict()

for simulation_index in range(number_of_simulations):

    print('Simulation', simulation_index)

    # The factorial design runs through each row of Yates array and translates
    # the value at each index to a corresponding parameter value in
    # design_variable_arr
    # if design_space_method == 'factorial_design':
    #     level_combination = yates_array[simulation_index, :]
    #     # Enumerate simplifies the code because the entries in yates_array can
    #     # directly be fed as indexes to the design parameters
    #     for it, j in enumerate(level_combination):  # Run through the row of levels from 0 to no_of_levels
    #         # IF WE SWITCH BETWEEN INDICES 1 AND -1, WE'LL ALWAYS BE GETTING "THE SECOND ENTRY" OF THE ARRAY.
    #         # WHAT I DID HERE ENSURES THAT THE LOWEST LEVEL CORRESPONDS TO ENTRY 0 OF THE ARRAY.
    #         if j == -1:
    #             shape_parameters[it] = design_variable_arr[int(j+no_of_levels/2), it]
    #         else:
    #             shape_parameters[it] = design_variable_arr[j, it]

    if design_space_method == 'monte_carlo':
        # If Monte Carlo, a random value is chosen with a uniform distribtion (NOTE: You can change the distribution)
        for parameter_index in range(number_of_parameters):
            shape_parameters[parameter_index] = np.random.uniform(decision_variable_range[0][parameter_index], decision_variable_range[1][parameter_index])

    elif design_space_method == 'monte_carlo_one_at_a_time':
            # If Monte Carlo, a random value is chosen with a uniform distribtion (NOTE: You can change the distribution)
            shape_parameters = nominal_parameters
            current_parameter = int(simulation_index/number_of_simulations_per_parameter)
            shape_parameters[current_parameter] = np.random.uniform(decision_variable_range[0][current_parameter], decision_variable_range[1][current_parameter])


    print('Parameters:', shape_parameters)
    parameters[simulation_index] = shape_parameters.copy()

    # Problem class is created
    current_capsule_entry_problem = ShapeOptimizationProblem(bodies,
                                                     termination_settings,
                                                     capsule_density,
                                                     simulation_start_epoch,
                                                     decision_variable_range)


    # NOTE: Propagator settings, termination settings, and initial_propagation_time are defined in the fitness function
    fitness = current_capsule_entry_problem.fitness(shape_parameters) # RIGHT NOW, FITNESS IS ALWAYS 0.0! MODIFY THE FUNCTION APPROPRIATELY
    print('Fitness:', fitness)
    objectives_and_constraints[simulation_index] = fitness

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = current_capsule_entry_problem.get_last_run_dynamics_simulator().state_history
    dependent_variable_history = current_capsule_entry_problem.get_last_run_dynamics_simulator().dependent_variable_history

    # Get output path
    subdirectory = '/DesignSpace_%s/Run_%s'%(design_space_method, simulation_index)

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

    # Delete data in aerodynamic coefficient database
    bodies.get_body( 'Capsule' ).aerodynamic_coefficient_interface.clear_data()

if write_results_to_file:
    subdirectory = '/DesignSpace_%s'%(design_space_method)
    output_path = current_dir + subdirectory
    save2txt(parameters, 'parameter_values.dat', output_path)
    save2txt(objectives_and_constraints, 'objectives_constraints.dat', output_path)


