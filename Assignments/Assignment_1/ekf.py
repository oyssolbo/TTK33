import numpy as np 
from typing import Callable

class EKF:
  def __init__(
        self, 
        continous_model : Callable, 
        x0              : np.ndarray, 
        P0              : np.ndarray,
        Qd              : np.ndarray,
        Rd              : np.ndarray,
        dt              : float
      ) -> None:
    """
    EKF for estimating a general continous function model

    Input-parameters:
      continous_model : function model to call
      x0              : initial state estimate
      P0              : initial covariance 
      Qd              : process noise - assumed discretized
      Rd              : measurement noise - assumed discretized 
      dt              : time-step [s]
    """

    self.continous_model : Callable = continous_model 
    
    self.x_hat : np.ndarray = x0
    self.P_hat : np.ndarray = P0

    # Assumning that the noise-matrices are already discretized
    self.Qd : np.ndarray = Qd 
    self.Rd : np.ndarray = Rd

    self.dt : float = dt
    self.n_states : int = self.x_hat.shape[0]


  def step(
        self, 
        u     : np.ndarray, 
        y     : np.ndarray = None
      ) -> tuple:
    """
    Runs a single step of the EKF

    Input-parameters:
      u     : control input to drive the function model
      y     : measurements. Assumed to be None if no measurements available 

    Output-parameters:
      x_hat : most recent state estimate
      P_hat : most recent covariance
    """
    # Assuming that an ERK1 - discretization is sufficient
    F = np.eye(self.n_states) + self.dt * self.continous_model.F_continous(x=self.x_hat, u=u) # ERK1 discretization

    # Prediction
    x_pred = self.x_hat + self.dt * self.continous_model.f_continous(x=self.x_hat, u=u) # ERK1 discretization
    P_pred = F @ self.P_hat @ F.T + self.Qd

    if y is not None:
      # Only correct the model if there are measurements
      y_pred = self.continous_model.g_continous(x=x_pred, u=u).ravel()

      # Kalman gain
      G = self.continous_model.G_continous(x=x_pred, u=u) # The measurement Jacobian will be equal for discrete and continous systems
      K = P_pred @ G.T @ np.linalg.inv(G @ P_pred @ G.T + self.Rd)

      # Correction 
      self.x_hat = x_pred + K @ (y - y_pred).reshape((2, 1))
      self.P_hat = (np.eye(self.n_states) - K @ G) @ P_pred 

    else:
      # If no measurements to correct, use the predictions 
      self.x_hat = x_pred
      self.P_hat = P_pred 

    return self.x_hat, self.P_hat
