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
This module contains the problem-specific classes and functions, which will be called by the main script where the
optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# Problem-specific imports
import LunarAscentUtilities as Util

# Tudatpy imports
import tudatpy
from tudatpy.data import save2txt
from tudatpy import constants
from tudatpy.numerical_simulation import propagation_setup
from tudatpy import numerical_simulation

###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class LunarAscentProblem:
    """
    Class to initialize, simulate and optimize the Lunar Ascent.
    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.
    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    constant_specific_impulse
    simulation_start_epoch
    dynamics_simulator
    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(thrust_parameters)
    """

    def __init__(self,
                 bodies: tudatpy.numerical_simulation.environment.SystemOfBodies,
                 termination_settings,
                 simulation_start_epoch: float,
                 vehicle_mass: float,
                 decision_variable_range):
        """
        Constructor for the LunarAscentProblem class.
        Parameters
        ----------
        bodies : tudatpy.numerical_simulation.environment_setup.SystemOfBodies
            System of bodies present in the simulation.
        integrator_settings : tudatpy.numerical_simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        propagator_settings : tudatpy.numerical_simulation.propagation_setup.propagator.MultiTypePropagatorSettings
            Propagator settings object.
        constant_specific_impulse : float
            Specific impulse of the vehicle that is kept constant during the propagation.
        simulation_start_epoch : float
            Epoch when the simulation begins [s].
        Returns
        -------
        none
        """
        # Set attributes
        self.bodies_function = lambda : bodies
        self.termination_settings_function = lambda :termination_settings
        self.simulation_start_epoch = simulation_start_epoch
        self.vehicle_mass = vehicle_mass
        self.decision_variable_range = decision_variable_range

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
                thrust_parameters: list) -> float:
        """
        Propagate the trajectory with the thrust parameters given as argument.
        This function uses the thrust parameters to set a new acceleration model, subsequently propagating the
        trajectory. The fitness, currently set to zero, can be computed here: it will be used during the optimization
        process.
        Parameters
        ----------
        thrust_parameters : list of floats
            List of thrust parameters.
        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """

        bodies = self.bodies_function()

        termination_settings = self.termination_settings_function( )
        dependent_variables_to_save = Util.get_dependent_variable_save_settings()

        propagator_settings = Util.get_propagator_settings(
            thrust_parameters,
            bodies,
            self.simulation_start_epoch,
            self.vehicle_mass,
            termination_settings,
            dependent_variables_to_save )
        propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            2.0, propagation_setup.integrator.CoefficientSets.rkdp_87)


        # Create simulation object and propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings )
        self.dynamics_simulator_function = lambda: dynamics_simulator

        # Add the objective and constraint values into the fitness vector
        fitness = 0.0
        return [fitness]