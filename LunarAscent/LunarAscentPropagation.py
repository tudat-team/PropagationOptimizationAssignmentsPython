"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Lunar Ascent
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module computes the dynamics of a Lunar ascent vehicle, according to a simple thrust guidance law.  This file propagates the dynamics
using a variety of integrator and propagator settings. For each run, the differences w.r.t. a benchmark propagation are
computed, providing a proxy for setting quality. The benchmark settings are currently defined semi-randomly, and are to be
analyzed/modified.

The propagtion starts with a small velocity close to the surface of the Moon, and an initial flight path angle of 90
degrees. Making (small) adjustments to this initial state is permitted if properly argued in the report.

The propagation is terminated as soon as one of the following conditions is met:

- Altitude > 100 km
- Altitude < 0 km
- Propagation time > 3600 s
- Vehicle mass < 2250 kg

This propagation assumes only point mass gravity by the Moon and thrust acceleration of the vehicle. Both the
translational dynamics and mass of the vehicle are propagated, using a fixed specific impulse.

The thrust is computed based on a constant thrust magnitude, and a variable thrust direction. The trust direction is defined
on a set of 5 nodes, spread evenly in time. At each node, a thrust angle theta is defined, which gives the angle between
the -z and y angles in the ascent vehicle's vertical frame (see Mooij, 1994, "The motion of a vehicle in a planetary
atmosphere" ). Between the nodes, the thrust is linearly interpolated. If the propagation goes beyond the bounds of
the nodes, the boundary value is used. The thrust profile is parameterized by the values of the vector thrust_parameters.
The thrust guidance is implemented in the LunarAscentThrustGuidance class in the LunarAscentUtilities.py file.

The entries of the vector 'thrust_parameters' contains the following:
- Entry 0: Constant thrust magnitude
- Entry 1: Constant spacing in time between nodes
- Entry 2-6: Thrust angle theta, at nodes 1-5 (in order)

Details on the outputs written by this file can be found:
- benchmark data: comments for 'generateBenchmarks' function
- results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"

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
from tudatpy import numerical_simulation
from tudatpy.math import interpolators
import tudatpy.util as util

# Problem-specific imports
import LunarAscentUtilities as Util
from LunarAscentProblem import LunarAscentProblem

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER
thrust_parameters = [15629.13262285292,
                     21.50263026822358,
                     -0.03344538412056863,
                     -0.06456210720352829,
                     0.3943447499535977,
                     0.5358478897251189,
                     -0.8607350478880107]
# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0.0  # s
# Vehicle settings
vehicle_mass = 4.7E3  # kg
vehicle_dry_mass = 2.25E3  # kg
constant_specific_impulse = 311.0  # s
# Fixed simulation termination settings
maximum_duration = constants.JULIAN_DAY  # s
termination_altitude = 100.0E3  # m

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Moon','Earth']
# Define coordinate system
global_frame_origin = 'Moon'
global_frame_orientation = 'ECLIPJ2000'

# Create body settings
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create,
    global_frame_origin,
    global_frame_orientation)

# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle object and add it to the existing system of bodies
bodies.create_empty_body('Vehicle')
# Set mass of vehicle
bodies.get_body('Vehicle').mass = vehicle_mass

# Create thrust model, with dummy settings, to be overridden when processing the thrust parameters
thrust_magnitude_settings = (
    propagation_setup.thrust.constant_thrust_magnitude( thrust_magnitude=0.0, specific_impulse=constant_specific_impulse ) )
environment_setup.add_engine_model(
    'Vehicle', 'MainEngine', thrust_magnitude_settings, bodies )
environment_setup.add_rotation_model(
    bodies, 'Vehicle', environment_setup.rotation_model.custom_inertial_direction_based(
        lambda time : np.array([1,0,0] ), global_frame_orientation, 'VehcleFixed' ) )

#######################################################################
### TERMINATION SETTINGS
###########################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                     maximum_duration,
                                                     termination_altitude,
                                                     vehicle_dry_mass)
################################
### DESIGN SPACE EXPLORATION ###
################################

# List of minimum and maximum values for each design parameter (thrust parameter)
decision_variable_range = \
    [[5.0E3, 10.0, -0.1, -0.5, -0.7, -1.0, -1.3],
     [20.0E3, 100.0, 0.1, 0.5 ,0.7, 1.0, 1.3]]

# NOTE TO STUDENTS: HERE YOU INPUT WHAT DESIGN SPACE EXPLORATION METHOD YOU USE
design_space_method = 'monte_carlo'

number_of_parameters = len(decision_variable_range[0])

# The number of Monte Carlo simulations is defined, as well as the seed which
# is passed to the MT19937 BitGenerator
if design_space_method == 'monte_carlo_one_at_a_time':
    number_of_simulations_per_parameter = 50
    number_of_simulations = number_of_parameters *  number_of_simulations_per_parameter
    nominal_parameters = [ 10000.0, 80.0, 1, 0.05, 0.25, 0.35, 0.5, 0.65 ]
    random_seed = 42 # ;)
    np.random.seed(random_seed) # Slightly outdated way of doing this, but works
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
#     design_variable_arr = np.zeros((no_of_levels, no_of_factors))
#
#     # Evenly distributed set of values between—and including—the minimum and maximum value
#     # defined earlier
#     for par in range(no_of_factors):
#         design_variable_arr[:, par] = np.linspace(decision_variable_range[0][par], decision_variable_range[1][par], no_of_levels, endpoint=True)
#         number_of_simulations = len(yates_array)

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
    #     for it, j in enumerate(level_combination): #Run through the row of levels from 0 to no_of_levels
    #         # IF WE SWITCH BETWEEN INDICES 1 AND -1, WE'LL ALWAYS BE GETTING "THE SECOND ENTRY" OF THE ARRAY.
    #         # WHAT I DID HERE ENSURES THAT THE LOWEST LEVEL CORRESPONDS TO ENTRY 0 OF THE ARRAY.
    #         if j == -1:
    #             thrust_parameters[it] = design_variable_arr[int(j+no_of_levels/2), it]
    #         else:
    #             thrust_parameters[it] = design_variable_arr[j, it]
    if design_space_method == 'monte_carlo':
        # For Monte Carlo and FFD, a separate loop exists
        for parameter_index in range(number_of_parameters):
            # If Monte Carlo, a random value is chosen with a uniform distribtion (NOTE: You can change the distribution)
            thrust_parameters[parameter_index] = np.random.uniform(decision_variable_range[0][parameter_index], decision_variable_range[1][parameter_index])
    elif design_space_method == 'monte_carlo_one_at_a_time':
            # If Monte Carlo, a random value is chosen with a uniform distribtion (NOTE: You can change the distribution)
            thrust_parameters = nominal_parameters
            current_parameter = int(simulation_index/number_of_simulations_per_parameter)
            thrust_parameters[current_parameter] = np.random.uniform(decision_variable_range[0][current_parameter], decision_variable_range[1][current_parameter])

    print('Thrust parameters:', thrust_parameters)
    parameters[simulation_index] = thrust_parameters.copy()

    # Problem class is created
    current_lunar_ascent_problem = LunarAscentProblem(bodies,
                                                      termination_settings,
                                                      simulation_start_epoch,
                                                      vehicle_mass,
                                                      decision_variable_range)


    # NOTE: Propagator settings, termination settings, and initial_propagation_time are defined in the fitness function
    fitness = current_lunar_ascent_problem.fitness(thrust_parameters)  # RIGHT NOW, FITNESS IS ALWAYS 0.0! MODIFY THE FUNCTION APPROPRIATELY
    print('Fitness:', fitness)
    objectives_and_constraints[simulation_index] = fitness

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = current_lunar_ascent_problem.get_last_run_dynamics_simulator().state_history
    dependent_variable_history = current_lunar_ascent_problem.get_last_run_dynamics_simulator().dependent_variable_history

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

if write_results_to_file:
    subdirectory = '/DesignSpace_%s'%(design_space_method)
    output_path = current_dir + subdirectory
    save2txt(parameters, 'parameter_values.dat', output_path)
    save2txt(objectives_and_constraints, 'objectives_constraints.dat', output_path)

