import numpy as np 

# Differential drive kinematics model found at 
# https://se.mathworks.com/help/robotics/ug/mobile-robot-kinematics-equations.html

class DifferentialDriveKinematicsModel:
  def f_continous(
        self, 
        x     : np.ndarray, 
        u     : np.ndarray
      ) -> np.ndarray:
    """
    Continous function model

    Input parameters:
      x = [x, y, theta]^T : states
      u = [v, omega]^T    : control inputs
    """

    theta = x[2]
    x_dot = np.array(
      [
        [np.cos(theta), 0],
        [np.sin(theta), 0],
        [0, 1]
      ], 
      dtype=np.float
    ) @ u 
    return x_dot.reshape((3, 1))

  def F_continous(
        self, 
        x     : np.ndarray, 
        u     : np.ndarray
      ) -> np.ndarray:
    """
    Continous Jacobian of state transition

    Input parameters:
      x = [x, y, theta]^T : states
      u = [v, omega]^T    : control inputs
    """

    v = u[0]
    theta = x[2]
    return np.array(
      [
        [0, 0, -np.sin(theta) * v],
        [0, 0, np.cos(theta) * v],
        [0, 0, 0]
      ],
      dtype=float
    )

  def g_continous(
        self, 
        x     : np.ndarray, 
        u     : np.ndarray
      ) -> np.ndarray:
    """
    Continous measurement model of the position. 
    This model does not use the 'control'-signals u, as it is only y = g(x) 

    Input parameters:
      x = [x, y, theta]^T : states
      u = [v, omega]^T    : control inputs
    """

    return x[:2].reshape((2, 1))

  def G_continous(
        self, 
        x     : np.ndarray, 
        u     : np.ndarray
      ) -> np.ndarray:
    """
    Continous Jacobian of measurement model.
    This model does not use the 'control'-signals u, as it is only y = g(x) 

    Input parameters:
      x = [x, y, theta]^T : states
      u = [v, omega]^T    : control inputs
    """

    return np.eye(2, 3)

