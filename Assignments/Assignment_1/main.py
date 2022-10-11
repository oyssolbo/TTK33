import numpy as np
import matplotlib.pyplot as plt
import math

import ekf 
import iekf
import ground_truth
import model

def main():
  # Initialization of memory
  num_states : int = 3
  num_iterations : int = 200#00

  show_covariance_matrices : bool = False # Not implemented atm, but could be nice in the future

  gt_states_array : np.ndarray = np.empty((num_states, num_iterations))
  ekf_state_estimates_array : np.ndarray = np.empty((num_states, num_iterations))
  iekf_state_estimates_array : np.ndarray = np.empty((num_states, num_iterations))
  if show_covariance_matrices:
    ekf_covariance_matrices_array : np.ndarray = np.empty((num_states, num_states, num_iterations))
    iekf_covariance_matrices_array : np.ndarray = np.empty((num_states, num_states, num_iterations))

  v_ref_array : np.ndarray = np.ones((1, num_iterations))
  omega_ref_array : np.ndarray = np.ones((1, num_iterations)) * (1 * np.pi / 180.0) 


  # Frequencies and timing 
  f_u : float = 100.0   # Velocity and angular rate
  f_x : float = 1       # Position

  dt = 1 / f_u


  # Initial values 
  x0 : np.ndarray = np.zeros((num_states, 1))
  P0 : np.ndarray = np.eye(num_states)


  # Noise - must be tuned
    # Process-noise
  q = [0.075, 0.1, 1 * dt * np.pi / 180.0]
    # Measurement noise
  r_pos = [1, 1]
  r_vel = [0.25 * dt, dt * 1 * np.pi / 180.0]

  Qd : np.ndarray = np.diag(q)
  Rd : np.ndarray = np.diag(r_pos)  


  differential_drive_model = model.DifferentialDriveKinematicsModel()
  EKF = ekf.EKF(
    continous_model=differential_drive_model,
    x0=x0,
    P0=P0,
    Qd=Qd,
    Rd=Rd,
    dt=dt  
  )
  IEKF = iekf.IEKF(
    continous_model=differential_drive_model,
    x0=x0,
    P0=P0,
    Qd=Qd,
    Rd=Rd,
    dt=dt,
    n_optimization=10  
  )
  gt = ground_truth.GroundTruth(
    continous_model=differential_drive_model,
    dt=dt,
    num_states=num_states
  )


  for iteration in range(num_iterations):
    v_ref = v_ref_array[0, iteration]
    omega_ref = omega_ref_array[0, iteration]

    x_exact, u_exact = gt.ground_truth(v_ref=v_ref, omega_ref=omega_ref)
    gt_states_array[:, iteration] = x_exact.ravel()

    # Injecting noise - assuming gaussian distributed noise
    # Assumption that the tuning is perfect, such that the values for Q and R
    # could be used in the noise-generation. This is of course not true in reality,
    # and one would have to tune the EKF 
    u_noise = u_exact + np.random.multivariate_normal(mean=np.zeros((len(r_vel))), cov=np.diag(r_vel))
    y_noise = None 
    if iteration % (int(f_u / f_x)) == 0:
      # Assuming that the measurement-model is exact 
      # Also assuming as above, that the noise-values used for the EKF R-matrix could be used
      y_noise = differential_drive_model.g_continous(x=x_exact, u=u_exact).ravel() + np.random.multivariate_normal(mean=np.zeros((len(r_pos))), cov=np.diag(r_pos)) 

    ekf_x_hat, ekf_P_hat = EKF.step(u=u_noise, y=y_noise)
    iekf_x_hat, iekf_P_hat = IEKF.step(u=u_noise, y=y_noise)

    ekf_state_estimates_array[:, iteration] = ekf_x_hat.ravel()
    iekf_state_estimates_array[:, iteration] = iekf_x_hat.ravel()


  # Plotting the ekf_errors as well to make it easier to vizualise
  fig, ax = plt.subplots(2)
  fig.canvas.manager.set_window_title("EKF estimates vs ground truth")

  x_gt = gt_states_array[0, :]
  y_gt = gt_states_array[1, :]
  theta_gt = gt_states_array[2, :]

  ekf_x_hat = ekf_state_estimates_array[0, :]
  ekf_y_hat = ekf_state_estimates_array[1, :]
  ekf_theta_hat = ekf_state_estimates_array[2, :]


  ax[0].plot(x_gt, y_gt, "g-", label="Ground truth")
  ax[0].plot(ekf_x_hat, ekf_y_hat, "r.", label="EKF estimate")
  ax[0].set_ylabel("y [m]")
  ax[0].set_xlabel("x [m]")
  ax[0].set_title("Position")

  ax[1].plot(theta_gt, "g-", label="Ground truth")
  ax[1].plot(ekf_theta_hat, "r.", label="EKF estimate")
  ax[1].set_ylabel("theta [rad]")
  ax[1].set_xlabel("k [0.01 s]")
  ax[1].set_title("Angle")


  fig, ax = plt.subplots(2)
  fig.canvas.manager.set_window_title("IEKF estimates vs ground truth")

  iekf_x_hat = ekf_state_estimates_array[0, :]
  iekf_y_hat = ekf_state_estimates_array[1, :]
  iekf_theta_hat = ekf_state_estimates_array[2, :]


  ax[0].plot(x_gt, y_gt, "g-", label="Ground truth")
  ax[0].plot(iekf_x_hat, iekf_y_hat, "r.", label="IEKF estimate")
  ax[0].set_ylabel("y [m]")
  ax[0].set_xlabel("x [m]")
  ax[0].set_title("Position")

  ax[1].plot(theta_gt, "g-", label="Ground truth")
  ax[1].plot(iekf_theta_hat, "r.", label="IEKF estimate")
  ax[1].set_ylabel("theta [rad]")
  ax[1].set_xlabel("k [0.01 s]")
  ax[1].set_title("Angle")


  fig, ax = plt.subplots(2)
  fig.canvas.manager.set_window_title("IEKF compared to EKF")

  ax[0].plot(ekf_x_hat, ekf_y_hat, "g.", iekf_x_hat, iekf_y_hat, "r.")
  ax[0].legend(["EKF", "IEKF"])
  # ax[0].plot(iekf_x_hat, iekf_y_hat, "r.", label="IEKF estimate")
  ax[0].set_ylabel("y [m]")
  ax[0].set_xlabel("x [m]")
  ax[0].set_title("Position")

  ax[1].plot(ekf_theta_hat, "g.", label="EKF estimate")
  ax[1].plot(iekf_theta_hat, "r.", label="IEKF estimate")
  ax[1].set_ylabel("theta [rad]")
  ax[1].set_xlabel("k [0.01 s]")
  ax[1].set_title("Angle")


  ekf_errors = np.subtract(gt_states_array, ekf_state_estimates_array) 
  ekf_error_fig, ekf_error_ax = plt.subplots(3)
  ekf_error_fig.canvas.manager.set_window_title("EKF errors")

  ekf_x_errors = ekf_errors[0, :]
  ekf_y_errors = ekf_errors[1, :]
  ekf_theta_errors = ekf_errors[2, :]

  ekf_error_ax[0].plot(ekf_x_errors, label="Errors in x")
  ekf_error_ax[0].set_ylabel("error x [m]")
  ekf_error_ax[0].set_xlabel("k [0.01 s]")
  ekf_error_ax[0].set_title("Error x-position")

  ekf_error_ax[1].plot(ekf_y_errors, label="Errors in y")
  ekf_error_ax[1].set_ylabel("error y [m]")
  ekf_error_ax[1].set_xlabel("k [0.01 s]")
  ekf_error_ax[1].set_title("Error y-position")
  
  ekf_error_ax[2].plot(ekf_theta_errors, label="Errors in theta")
  ekf_error_ax[2].set_ylabel("error theta [rad]")
  ekf_error_ax[2].set_xlabel("k [0.01 s]")
  ekf_error_ax[2].set_title("Error theta-angle")

  print("EKF")
  print("Number of x-estimates with an error greater than 5 m: ", len(np.where(np.abs(ekf_x_errors) > 5)[0]))
  print("Number of x-estimates with an error greater than 10 m: ", len(np.where(np.abs(ekf_x_errors) > 10)[0]))
  print("Number of x-estimates with an error greater than 20 m: ", len(np.where(np.abs(ekf_x_errors) > 20)[0]))
  print("Number of x-estimates with an error greater than 40 m: ", len(np.where(np.abs(ekf_x_errors) > 40)[0]))


  iekf_errors = np.subtract(gt_states_array, iekf_state_estimates_array) 
  iekf_error_fig, iekf_error_ax = plt.subplots(3)
  iekf_error_fig.canvas.manager.set_window_title("IEKF errors")

  iekf_x_errors = iekf_errors[0, :]
  iekf_y_errors = iekf_errors[1, :]
  iekf_theta_errors = iekf_errors[2, :]

  iekf_error_ax[0].plot(iekf_x_errors, label="Errors in x")
  iekf_error_ax[0].set_ylabel("error x [m]")
  iekf_error_ax[0].set_xlabel("k [0.01 s]")
  iekf_error_ax[0].set_title("Error x-position")

  iekf_error_ax[1].plot(iekf_y_errors, label="Errors in y")
  iekf_error_ax[1].set_ylabel("error y [m]")
  iekf_error_ax[1].set_xlabel("k [0.01 s]")
  iekf_error_ax[1].set_title("Error y-position")
  
  iekf_error_ax[2].plot(iekf_theta_errors, label="Errors in theta")
  iekf_error_ax[2].set_ylabel("error theta [rad]")
  iekf_error_ax[2].set_xlabel("k [0.01 s]")
  iekf_error_ax[2].set_title("Error theta-angle")

  print("IEKF")
  print("Number of x-estimates with an error greater than 5 m: ", len(np.where(np.abs(iekf_x_errors) > 5)[0]))
  print("Number of x-estimates with an error greater than 10 m: ", len(np.where(np.abs(iekf_x_errors) > 10)[0]))
  print("Number of x-estimates with an error greater than 20 m: ", len(np.where(np.abs(iekf_x_errors) > 20)[0]))
  print("Number of x-estimates with an error greater than 40 m: ", len(np.where(np.abs(iekf_x_errors) > 40)[0]))

  plt.show()



if __name__ == '__main__':
  main()

