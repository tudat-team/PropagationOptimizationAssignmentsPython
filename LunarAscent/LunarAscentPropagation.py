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
#
# import sys
# sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.data import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice as spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import LunarAscentUtilities as Util

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
# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = True

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

# Set number of models
number_of_models = 5

# Initialize dictionary to store the results of the simulation
simulation_results = dict()

# Set the interpolation step at which different runs are compared
output_interpolation_step = 1.0  # s

for model_test in range(number_of_models):
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

    # For case 4, the ephemeris of the Moon is generated by solving the 2-body problem of the Earth and the Moon
    # (in the other cases, the ephemeris of the Moon from SPICE takes into account all the perturbations)
    if (model_test == 4):
        effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Moon') + \
                                            spice_interface.get_body_gravitational_parameter('Earth')
        body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
            'Moon', simulation_start_epoch, effective_gravitational_parameter, 'Earth', global_frame_orientation)

    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create vehicle object and add it to the existing system of bodies
    bodies.create_empty_body('Vehicle')
    # Set mass of vehicle
    bodies.get_body('Vehicle').mass = vehicle_mass

    # Create thrust model, with dummy settings, to be overridden when processing the thrust parameters
    thrust_magnitude_settings = (
        propagation_setup.thrust.constant_thrust_magnitude(thrust_magnitude=0.0,
                                                           specific_impulse=constant_specific_impulse))
    environment_setup.add_engine_model(
        'Vehicle', 'MainEngine', thrust_magnitude_settings, bodies)
    environment_setup.add_rotation_model(
        bodies, 'Vehicle', environment_setup.rotation_model.custom_inertial_direction_based(
            lambda time: np.array([1, 0, 0]), global_frame_orientation, 'VehcleFixed'))

    ###########################################################################
    # CREATE PROPAGATOR SETTINGS ##############################################
    ###########################################################################

    # Retrieve termination settings
    termination_settings = Util.get_termination_settings(simulation_start_epoch,
                                                         maximum_duration,
                                                         termination_altitude,
                                                         vehicle_dry_mass)
    # Retrieve dependent variables to save
    dependent_variables_to_save = Util.get_dependent_variable_save_settings()
    # Check whether there is any
    are_dependent_variables_to_save = False if not dependent_variables_to_save else True

    propagator_settings = Util.get_propagator_settings(
        thrust_parameters,
        bodies,
        simulation_start_epoch,
        vehicle_mass,
        termination_settings,
        dependent_variables_to_save,
        current_propagator=propagation_setup.propagator.cowell,
        model_choice = model_test )

    # Create integrator settings
    propagator_settings.integrator_settings = Util.get_integrator_settings(
        0, 0, 0, simulation_start_epoch)

    # Create Lunar Ascent Problem object
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    # Save results to a dictionary
    simulation_results[model_test] = [state_history, dependent_variable_history]

    # Get output path
    if model_test == 0:
        subdirectory = '/NominalCase/'
    else:
        subdirectory = '/Model_' + str(model_test) + '/'

    # Decide if output writing is required
    if write_results_to_file:
        output_path = current_dir + subdirectory
    else:
        output_path = None

    # If desired, write output to a file
    if write_results_to_file:
        save2txt(state_history, 'state_history.dat', output_path)
        save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

"""
NOTE TO STUDENTS
The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
or 1 (dependent variables).
You can use this dictionary to make all the cross-comparison that you deem necessary. The code below currently compares
every case with respect to the "nominal" one.
"""
# Compare all the model settings with the nominal case
for model_test in range(1, number_of_models):
    # Get output path
    output_path = current_dir + '/Model_' + str(model_test) + '/'

    # Set time limits to avoid numerical issues at the boundaries due to the interpolation
    nominal_state_history = simulation_results[0][0]
    nominal_dependent_variable_history = simulation_results[0][1]
    nominal_times = list(nominal_state_history.keys())

    # Retrieve current state and dependent variable history
    current_state_history = simulation_results[model_test][0]
    current_dependent_variable_history = simulation_results[model_test][1]
    current_times = list(current_state_history.keys())

    # Get limit times at which both histories can be validly interpolated
    interpolation_lower_limit = max(nominal_times[3],current_times[3])
    interpolation_upper_limit = min(nominal_times[-3],current_times[-3])

    # Create vector of epochs to be compared (boundaries are referred to the first case)
    unfiltered_interpolation_epochs = np.arange(current_times[0], current_times[-1], output_interpolation_step)
    unfiltered_interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n <= interpolation_upper_limit]
    interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

    # Compare state history
    state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                       simulation_results[0][0],
                                                       interpolation_epochs,
                                                       output_path,
                                                       'state_difference_wrt_nominal_case.dat')
    # Compare dependent variable history
    dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                    simulation_results[0][1],
                                                                    interpolation_epochs,
                                                                    output_path,
                                                                    'dependent_variable_difference_wrt_nominal_case.dat')


