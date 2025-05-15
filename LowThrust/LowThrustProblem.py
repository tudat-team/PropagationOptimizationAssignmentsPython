'''
Copyright (c) 2010-2021, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Low Thrust
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

# Problem-specific imports
import LowThrustUtilities as Util

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.numerical_simulation import propagation_setup
from tudatpy import numerical_simulation
from tudatpy.math import interpolators
from tudatpy.trajectory_design import shape_based_thrust
from tudatpy.trajectory_design import transfer_trajectory

###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class LowThrustProblem:
    """
    Class to initialize, simulate and optimize the Low Thrust trajectory.
    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.
    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    specific_impulse
    minimum_mars_distance
    time_buffer
    perform_propagation
    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(trajectory_parameters)
    get_hodographic_shaping()
    """

    def __init__(self,
                 bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                 minimum_mars_distance: float,
                 time_buffer: float,
                 vehicle_mass: float,
                 decision_variable_range,
                 perform_propagation: bool = True):
        """
        Constructor for the LowThrustProblem class.
        Parameters
        ----------
        bodies : tudatpy.numerical_simulation.environment.SystemOfBodies,
            System of bodies present in the simulation.
        minimum_mars_distance : float
            Minimum distance from Mars at which the propagation stops.
        time_buffer : float
            Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
        perform_propagation : bool (default: True)
            If true, the propagation is performed.
        Returns
        -------
        none
        """
        # Copy arguments as attributes
        self.bodies_function = lambda : bodies
        self.minimum_mars_distance = minimum_mars_distance
        self.time_buffer = time_buffer
        self.vehicle_mass = vehicle_mass
        self.decision_variable_range = decision_variable_range
        self.perform_propagation = perform_propagation

    def get_bounds(self):

        return self.decision_variable_range

    def get_last_run_propagated_cartesian_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution()

    def get_last_run_propagated_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, not converted to Cartesian state
        (i.e. in the actual formulation that was used during the numerical integration).
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_equations_of_motion_numerical_solution_raw()

    def get_last_run_dependent_variable_history(self) -> dict:
        """
        Returns the full history of the dependent variables.
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function( ).get_dependent_variable_history()

    def get_last_run_dynamics_simulator(self) -> tudatpy.numerical_simulation.SingleArcSimulator:
        """
        Returns the dynamics simulator object.
        Parameters
        ----------
        none
        Returns
        -------
        tudatpy.numerical_simulation.SingleArcSimulator
        """
        return self.dynamics_simulator_function( )

    def fitness(self,
                trajectory_parameters) -> float:
        """
        Propagate the trajectory using the hodographic method with the parameters given as argument.
        This function uses the trajectory parameters to create a new hodographic shaping object, from which a new
        thrust acceleration profile is extracted. Subsequently, the trajectory is propagated numerically, if desired.
        The fitness, currently set to zero, can be computed here: it will be used during the optimization process.
        Parameters
        ----------
        trajectory_parameters : list of floats
            List of trajectory parameters to optimize.
        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        # Create hodographic shaping object
        trajectory_parameters[2]=round(trajectory_parameters[2])

        bodies = self.bodies_function()
        trajectory_object = Util.create_hodographic_trajectory(trajectory_parameters, bodies)

        # Propagate trajectory only if required
        if self.perform_propagation:

            # Retrieve termination settings
            termination_settings = Util.get_termination_settings(trajectory_parameters,
                                                                 self.minimum_mars_distance,
                                                                 self.time_buffer)
            # Retrieve dependent variables to save
            dependent_variables_to_save = Util.get_dependent_variable_save_settings()
            # Check whether there is any
            are_dependent_variables_to_save = False if not dependent_variables_to_save else True

            # Create propagator settings for benchmark (Cowell)
            initial_propagation_time = Util.get_trajectory_initial_time(
                trajectory_parameters, self.time_buffer)
            propagator_settings = Util.get_propagator_settings(
                trajectory_parameters,
                bodies,
                initial_propagation_time,
                self.vehicle_mass,
                termination_settings,
                dependent_variables_to_save,
                current_propagator=propagation_setup.propagator.cowell )

            propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
                86400.0, propagation_setup.integrator.CoefficientSets.rkdp_87)


            # Create simulation object and propagate dynamics
            dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                bodies, propagator_settings )

            self.dynamics_simulator_function = lambda : dynamics_simulator

        # Add the objective and constraint values into the fitness vector
        fitness = trajectory_object.delta_v_per_leg[ 0 ]
        return [fitness]
