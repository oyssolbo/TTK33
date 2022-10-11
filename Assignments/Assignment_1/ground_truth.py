import numpy as np
from typing import Callable

class GroundTruth:
  def __init__(
        self, 
        continous_model : Callable, 
        dt              : float, 
        num_states      : int
      ) -> None:
    """
    Generates the ground-truth data

    Uses the continous model to generate some of the data, and therefore
    assumes the model to be exact - which it in reality will not
    """
    self.continous_model : Callable = continous_model
    self.dt : float = dt

    # Parameters for the reference-models (randomly chosen)
      # Velocity 
    self.tau_v : float = 2.5
    self.k_v : float = 1

      # Angular rate
    self.tau_omega : float = 0.5
    self.k_omega : float = 1 

    # Initial values
    self.v : float = 0
    self.omega : float = 0
    self.x : np.ndarray = np.zeros((num_states, 1))


  def _velocity_reference_model(
        self, 
        v     : float, 
        v_ref : float
      ) -> float:
    """
    First-order model generating the velocities
    """
    v_dot = (1 / self.tau_v) * (-v + self.k_v * v_ref)
    return v + self.dt * v_dot

  def _angular_rate_reference_model(
        self, 
        omega     : float, 
        omega_ref : float
      ) -> float:
    """
    First-order model generating the velocities
    """
    omega_dot = (1 / self.tau_v) * (-omega + self.k_v * omega_ref)
    return omega + self.dt * omega_dot

  def ground_truth(
        self, 
        v_ref     : float, 
        omega_ref : float
      ) -> np.ndarray:
    """
    Calculates the ground truth based on the desired velocity and 
    the desired angular rate
    """
    
    self.v = self._velocity_reference_model(v=self.v, v_ref=v_ref)
    self.omega = self._angular_rate_reference_model(omega=self.omega, omega_ref=omega_ref)

    u = np.array((self.v, self.omega)).T
    x_dot = self.continous_model.f_continous(x=self.x, u=u)
    
    self.x = self.x + self.dt * x_dot
    return self.x, u

