import numpy as np 
from typing import Callable

class IEKF:
  def __init__(
        self, 
        continous_model : Callable, 
        x0              : np.ndarray, 
        P0              : np.ndarray, 
        Qd              : np.ndarray, 
        Rd              : np.ndarray, 
        dt              : float,
        n_optimization  : int
      ) -> None:
    """
    Iterated EKF for estimating a general continous function model

    Input-parameters:
      continous_model : function model to call
      x0              : initial state estimate
      P0              : initial covariance 
      Qd              : process noise - assumed discretized
      Rd              : measurement noise - assumed discretized 
      dt              : time-step [s]
      n_optimization  : number of iterations to optimize the estimate
    """  
    self.continous_model : Callable = continous_model 

    # Assumning that the noise-matrices are already discretized
    self.Qd : np.ndarray = Qd 
    self.Rd : np.ndarray = Rd

    self.dt : float = dt
    self.n_states : int = x0.shape[0]

    self.x_op : np.ndarray = x0
    self.P_op : np.ndarray = P0

    self.n_optimization : int = n_optimization


  def step(
        self, 
        u     : np.ndarray, 
        y     : np.ndarray = None
      ) -> tuple:
    """
    Runs a single step of the IEKF, by iterating the EKF-estimate over @p n_optimization

    Will only iterate if there are measurements

    Input-parameters:
      u     : control input to drive the function model
      y     : measurements. Assumed to be None if no measurements available 

    Output-parameters:
      x_op  : most recent state estimate
      P_op  : most recent covariance
    """
    n_optimization = self.n_optimization
    if y is None:
      n_optimization = 1

    for iteration in range(n_optimization):
      # Assuming that an ERK1 - discretization is sufficient
      F = np.eye(self.n_states) + self.dt * self.continous_model.F_continous(x=self.x_op, u=u) # ERK1 discretization

      # Prediction
      x_pred = self.x_op + self.dt * self.continous_model.f_continous(x=self.x_op, u=u) # ERK1 discretization
      P_pred = F @ self.P_op @ F.T + self.Qd

      if y is not None:
        # Only correct the model if there are measurements
        y_op = self.continous_model.g_continous(x=self.x_op, u=u).ravel()

        # Kalman gain
        G = self.continous_model.G_continous(x=x_pred, u=u) # The measurement Jacobian will be equal for discrete and continous systems
        K = P_pred @ G.T @ np.linalg.inv(G @ P_pred @ G.T + self.Rd)

        # Correction 
        self.x_op = x_pred + K @ (y - y_op - (G @ (x_pred - self.x_op)).ravel()).reshape((2, 1))
        self.P_op = (np.eye(self.n_states) - K @ G) @ P_pred

      else:
        # If no measurements to correct, use the predictions 
        self.x_op = x_pred
        self.P_op = P_pred 

    return self.x_op, self.P_op

