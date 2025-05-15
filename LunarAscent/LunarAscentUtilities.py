'''
Copyright (c) 2010-2021, Delft University of Technology
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

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.interface import spice as spice_interface
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import environment
from tudatpy import numerical_simulation
from tudatpy.astro import element_conversion
from tudatpy.math import interpolators

###########################################################################
# PROPAGATION SETTING UTILITIES ###########################################
###########################################################################

def get_initial_state(simulation_start_epoch: float,
                      bodies: tudatpy.numerical_simulation.environment.SystemOfBodies) -> np.ndarray:
    """
    Converts the initial state to inertial coordinates.

    The initial state is expressed in Moon-centered spherical coordinates.
    These are first converted into Moon-centered cartesian coordinates,
    then they are finally converted in the global (inertial) coordinate
    system.

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    initial_state_inertial_coordinates : np.ndarray
        The initial state of the vehicle expressed in inertial coordinates.
    """
    # Set initial spherical elements.
    radius = spice_interface.get_average_radius('Moon') + 100.0
    latitude = np.deg2rad(0.6875)
    longitude = np.deg2rad(23.4333)
    speed = 10.0
    flight_path_angle = np.deg2rad(89.0)
    heading_angle = np.deg2rad(90.0)

    # Convert spherical elements to body-fixed cartesian coordinates
    initial_cartesian_state_body_fixed = element_conversion.spherical_to_cartesian_elementwise(
        radius, latitude,  longitude, speed, flight_path_angle, heading_angle)
    # Get rotational ephemerides of the Moon
    moon_rotational_model = bodies.get_body('Moon').rotation_model
    # Transform the state to the global (inertial) frame
    initial_state_inertial_coordinates = environment.transform_to_inertial_orientation(
        initial_cartesian_state_body_fixed,
        simulation_start_epoch,
        moon_rotational_model)

    return initial_state_inertial_coordinates


def get_termination_settings(simulation_start_epoch: float,
                             maximum_duration: float,
                             termination_altitude: float,
                             vehicle_dry_mass: float) \
        -> tudatpy.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (one day)
    - lower and upper altitude boundaries (0-100 km)
    - fuel run-out

    Parameters
    ----------
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    maximum_duration : float
        Maximum duration of the simulation [s].
    termination_altitude : float
        Maximum altitude [m].
    vehicle_dry_mass : float
        Dry mass of the spacecraft [kg].

    Returns
    -------
    hybrid_termination_settings : tudatpy.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    time_termination_settings = propagation_setup.propagator.time_termination(
        simulation_start_epoch + maximum_duration,
        terminate_exactly_on_final_condition=False
    )
    # Altitude
    upper_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Vehicle', 'Moon'),
        limit_value=termination_altitude,
        use_as_lower_limit=False,
        terminate_exactly_on_final_condition=False
    )
    lower_altitude_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.altitude('Vehicle', 'Moon'),
        limit_value=0.0,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )
    # Vehicle mass
    mass_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.body_mass('Vehicle'),
        limit_value=vehicle_dry_mass,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False
    )
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 upper_altitude_termination_settings,
                                 lower_altitude_termination_settings,
                                 mass_termination_settings]
    # Create termination settings object
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings


# NOTE TO STUDENTS: this function can be modified to save more/less dependent variables.
def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include:
    - the altitude wrt the Moon
    - the relative speed wrt the Moon
    - the flight path angle of the vehicle

    Parameters
    ----------
    none

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.altitude('Vehicle', 'Moon'),
                                   propagation_setup.dependent_variable.relative_speed('Vehicle', 'Moon'),
                                   propagation_setup.dependent_variable.flight_path_angle('Vehicle', 'Moon'),
                                   propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.thrust_acceleration_type, 'Vehicle', 'Vehicle')]
    return dependent_variables_to_save

def get_propagator_settings(thrust_parameters,
                            bodies,
                            simulation_start_epoch,
                            vehicle_initial_mass,
                            termination_settings,
                            dependent_variables_to_save ):
    """
    Creates the propagator settings.

    This function creates the propagator settings for translational motion and mass, for the given simulation settings
    Note that, in this function, the thrust_parameters are used to update the engine model and rotation model of the
    vehicle. The propagator settings that are returned as output of this function are not yet usable: they do not
    contain any integrator settings, which should be set at a later point by the user

    Parameters
    ----------
    thrust_parameters : list[ float ]
        List of free parameters for the thrust model, which will be used to update the vehicle properties such that
        the new thrust/magnitude direction are used. The meaning of the parameters in this list is stated at the
        start of the *Propagation.py file
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.
    vehicle_initial_mass : float
        Mass of the vehicle to be used at the initial time
    termination_settings : tudatpy.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object to be used
    dependent_variables_to_save : list[tudatpy.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    current_propagator : tudatpy.numerical_simulation.propagation_setup.propagator.TranslationalPropagatorType
        Type of propagator to be used for translational dynamics

    Returns
    -------
    propagator_settings : tudatpy.numerical_simulation.propagation_setup.integrator.MultiTypePropagatorSettings
        Propagator settings to be provided to the dynamics simulator.
    """

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Moon']

    # Define accelerations acting on vehicle
    thrust_settings = set_thrust_acceleration_model_from_parameters(
        thrust_parameters,
        bodies,
        simulation_start_epoch)
    acceleration_settings_on_vehicle = {
        'Moon': [propagation_setup.acceleration.point_mass_gravity()],
        'Vehicle': [thrust_settings]
    }

    # Create acceleration models.
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state
    initial_state = get_initial_state(simulation_start_epoch, bodies)

    # Create propagation settings for the translational dynamics
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        None,
        termination_settings,
        propagation_setup.propagator.cowell,
        output_variables=dependent_variables_to_save)

    # Create mass rate model
    mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass_rate.from_thrust()]}
    mass_rate_models = propagation_setup.create_mass_rate_models(bodies,
                                                                 mass_rate_settings_on_vehicle,
                                                                 acceleration_models)

    # Create mass propagator settings
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_models,
                                                                 np.array([vehicle_initial_mass]),
                                                                 simulation_start_epoch,
                                                                 None,
                                                                 termination_settings)

    # Create multi-type propagation settings list
    propagator_settings_list = [translational_propagator_settings,
                                mass_propagator_settings]

    # Create multi-type propagation settings object for translational dynamics and mass.
    # NOTE: these are not yet 'valid', as no integrator settings are defined yet
    propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                 None,
                                                                 simulation_start_epoch,
                                                                 termination_settings,
                                                                 dependent_variables_to_save)

    return propagator_settings


###########################################################################
# GUIDANCE/THRUST UTILITIES ###############################################
###########################################################################

class LunarAscentThrustGuidance:
    """
    Class that defines and updates the thrust guidance of the Lunar Ascent problem at each time step.

    Attributes
    ----------
    vehicle_body
    initial_time
    parameter_vector

    Methods
    -------
    get_current_thrust_direction(time)
    """

    def __init__(self,
                 vehicle_body: str,
                 initial_time: float,
                 parameter_vector: list):
        """
        Constructor of the LunarAscentThrustGuidance class.

        Parameters
        ----------
        vehicle_body : str
            Name of the vehicle to apply the thrust guidance to.
        initial_time: float
            Initial time of the simulation [s].
        parameter_vector : list
            List of thrust parameters to retrieve the thrust guidance from.

        Returns
        -------
        none
        """
        # Set arguments as attributes
        self.vehicle_body = vehicle_body
        self.initial_time = initial_time
        self.parameter_vector = parameter_vector
        self.time_interval = parameter_vector[1]
        # Prepare dictionary for thrust angles
        self.thrust_angle_dict = {}
        self.thrust_angle_derivative_dict = list()

        # Initialize time
        current_time = initial_time

        # Loop over nodes
        for i in range(len(parameter_vector) - 2):

            # Store time as key, thrust angle as value
            self.thrust_angle_dict[current_time] = parameter_vector[i + 2]

            # Set thrust angle derivative as central difference from previous and next node, except on first and last node
            # where derivative is set to 0
            if (i == 0 or i + 2 == (len(parameter_vector) - 1)):
                self.thrust_angle_derivative_dict.append(0.0)
            else:
                self.thrust_angle_derivative_dict.append(
                    (parameter_vector[i + 3] - parameter_vector[i + 1]) / (2.0 * self.time_interval))

            # Increase time
            current_time += self.time_interval

        # Add final node 1 day after the fine node, and set its thrust value equal to that on the final node,
        # and teh angle derivative to 0.
        self.thrust_angle_dict[current_time + 86400.0 ] = parameter_vector[-1]
        self.thrust_angle_derivative_dict.append( 0.0 )

        # Create interpolator settings
        interpolator_settings = interpolators.hermite_spline_interpolation(
            boundary_interpolation=interpolators.use_boundary_value)

        # Create the interpolator between nodes and set it as attribute
        self.thrust_angle_interpolator = interpolators.create_one_dimensional_scalar_interpolator(
            self.thrust_angle_dict, interpolator_settings, self.thrust_angle_derivative_dict )

    def get_current_thrust_direction(self,
                                     time: float) -> np.array:
        """
        Retrieves the direction of the thrust in the inertial frame.

        This function is needed to get the thrust direction at each time step of the propagation, based on the thrust
        parameters provided.

        Parameters
        ----------
        time : float
            Current time.

        Returns
        -------
        thrust_inertial_frame : np.array
            Thrust direction expressed in the inertial frame.
        """
        # Interpolate with time
        angle = self.thrust_angle_interpolator.interpolate(time)
        # Set thrust vector in vertical frame
        thrust_direction_vertical_frame = np.array([[0, np.sin(angle), - np.cos(angle)]]).T
        # Update flight conditions (this is needed to let tudat know to update all variables)
        self.vehicle_body.flight_conditions.update_conditions(time)

        # Get aerodynamic angle calculator
        aerodynamic_angle_calculator = self.vehicle_body.flight_conditions.aerodynamic_angle_calculator
        # Retrieve rotation matrix from vertical to inertial frame from the aerodynamic angle calculator
        vertical_to_inertial_frame = aerodynamic_angle_calculator.get_rotation_matrix_between_frames(
            environment.vertical_frame,
            environment.inertial_frame)
        # Compute the thrust in the inertial frame
        thrust_inertial_frame = np.dot(vertical_to_inertial_frame,
                                       thrust_direction_vertical_frame)
        return thrust_inertial_frame

def set_thrust_acceleration_model_from_parameters(thrust_parameters: list,
                                                  bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                                                  initial_time: float) -> \
        tudatpy.numerical_simulation.propagation_setup.acceleration.ThrustAccelerationSettings:
    """
    Creates the thrust acceleration models from the LunarAscentThrustGuidance class and sets it in the propagator.

    Parameters
    ----------
    thrust_parameters : list of floats
        List of thrust parameters.
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    initial_time : float
        The start time of the simulation in seconds.

    Returns
    -------
    tudatpy.numerical_simulation.propagation_setup.acceleration.ThrustAccelerationSettings
        Thrust acceleration settings object.
    """
    # Create Thrust Guidance object
    thrust_guidance = LunarAscentThrustGuidance(bodies.get_body('Vehicle'),
                                                initial_time,
                                                thrust_parameters)
    # Retrieves thrust functions
    thrust_direction_function = thrust_guidance.get_current_thrust_direction

    # Retrieve engine model and reset thrust level
    main_engine_model = bodies.get_body('Vehicle').system_models.get_engine_model( 'MainEngine' )
    main_engine_model.thrust_magnitude_calculator.constant_thrust_magnitude = thrust_parameters[0]

    # Set thrust functions in the acceleration model
    vehicle_rotation_model = bodies.get_body('Vehicle').rotation_model
    vehicle_rotation_model.inertial_body_axis_calculator.inertial_body_axis_direction_function = thrust_direction_function

    # Create thrust acceleration settings
    acceleration_settings = propagation_setup.acceleration.thrust_from_engine( 'MainEngine' )

    # Create and return thrust acceleration settings
    return acceleration_settings


###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################


# NOTE TO STUDENTS: THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
def generate_benchmarks(benchmark_step_size: float,
                        bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings:
                        tudatpy.numerical_simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                        are_dependent_variables_present: bool,
                        output_path: str = None):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired, 
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    bodies : tudatpy.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    thrust_parameters
        List that represents the thrust parameters for the spacecraft.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """
    ### CREATION OF THE TWO BENCHMARKS ###
    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size  # s
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        first_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

    print('Running first benchmark...')
    first_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings)

    # Create integrator settings for the second benchmark in the same way
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        second_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = False

    print('Running second benchmark...')
    second_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings)

    ### WRITE BENCHMARK RESULTS TO FILE ###
    # Retrieve state history
    first_benchmark_states = first_dynamics_simulator.state_history
    second_benchmark_states = second_dynamics_simulator.state_history
    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, 'benchmark_1_states.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_dynamics_simulator.dependent_variable_history
        second_benchmark_dependent_variable = second_dynamics_simulator.dependent_variable_history
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables.dat',  output_path)
            save2txt(second_benchmark_dependent_variable,  'benchmark_2_dependent_variables.dat',  output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list


def compare_models(first_model: dict,
                   second_model: dict,
                   interpolation_epochs: np.ndarray,
                   output_path: str,
                   filename: str) -> dict:
    """
    It compares the results of two runs with different model settings.
    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.
    Parameters
    ----------
    first_model : dict
        State (or dependent variable history) from the first run.
    second_model : dict
        State (or dependent variable history) from the second run.
    interpolation_epochs : np.ndarray
        Vector of epochs at which the two runs are compared.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.
    Returns
    -------
    model_difference : dict
        Interpolated difference between the two simulations' state (or dependent variable) history.
    """
    # Create interpolator settings
    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.use_boundary_value)
    # Create 8th-order Lagrange interpolator for both cases
    first_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_model, interpolator_settings)
    second_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        second_model, interpolator_settings)
    # Calculate the difference between the first and second model at specific epochs
    model_difference = {epoch: second_interpolator.interpolate(epoch) - first_interpolator.interpolate(epoch)
                        for epoch in interpolation_epochs}
    # Write results to files
    if output_path is not None:
        save2txt(model_difference,
                 filename,
                 output_path)
    # Return the model difference
    return model_difference

def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark,  interpolators.lagrange_interpolation(8))
    # Calculate the difference between the benchmarks
    print('Calculating benchmark differences...')
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                             second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference
