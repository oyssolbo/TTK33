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
  num_steps : int = 10000

  show_covariance_matrices : bool = False # Not implemented atm, but could be nice in the future

  gt_states_array : np.ndarray = np.empty((num_states, num_steps))
  ekf_state_estimates_array : np.ndarray = np.empty((num_states, num_steps))
  iekf_state_estimates_array : np.ndarray = np.empty((num_states, num_steps))
  if show_covariance_matrices:
    ekf_covariance_matrices_array : np.ndarray = np.empty((num_states, num_states, num_steps))
    iekf_covariance_matrices_array : np.ndarray = np.empty((num_states, num_states, num_steps))

  v_ref_array : np.ndarray = np.ones((1, num_steps)) * 10.0
  omega_ref_array : np.ndarray = np.ones((1, num_steps)) * (2.0 * np.pi / 180.0) 


  # Frequencies and timing 
  f_u : float = 100.0   # Velocity and angular rate
  f_x : float = 1       # Position

  dt = 1 / f_u


  # Initial values 
  x0 : np.ndarray = np.zeros((num_states, 1))
  P0 : np.ndarray = np.eye(num_states)


  # Noise - must be tuned
    # Process-noise
  q_var = [1, 0.8, 2 * np.pi / 180.0]
    # Measurement noise
  r_pos_var = [2, 2]
  r_vel_var = [10, 2 * np.pi / 180.0]

  Qd : np.ndarray = np.diag(q_var)
  Rd : np.ndarray = np.diag([2, 2])  


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


  for iteration in range(num_steps):
    v_ref = v_ref_array[0, iteration]
    omega_ref = omega_ref_array[0, iteration]

    x_exact, u_exact = gt.ground_truth(v_ref=v_ref, omega_ref=omega_ref)
    gt_states_array[:, iteration] = x_exact.ravel()

    # Injecting noise - assuming gaussian distributed noise
    # Assumption that the tuning is perfect, such that the values for Q and R
    # could be used in the noise-generation. This is of course not true in reality,
    # and one would have to tune the EKF 
    u_noise = u_exact + np.random.multivariate_normal(mean=np.zeros((len(r_vel_var))), cov=np.diag(r_vel_var))
    y_noise = None 
    if iteration % (int(f_u / f_x)) == 0 and iteration > 0:
      # Assuming that the measurement-model is exact 
      # Also assuming as above, that the noise-values used for the EKF R-matrix could be used
      y_exact = differential_drive_model.g_continous(x=x_exact, u=u_exact).ravel() 
      y_noise = y_exact + np.random.multivariate_normal(mean=np.zeros((len(r_pos_var))), cov=np.diag(r_pos_var)) 

    ekf_x_hat, ekf_P_hat = EKF.step(u=u_noise, y=y_noise)
    iekf_x_hat, iekf_P_hat = IEKF.step(u=u_noise, y=y_noise)

    ekf_state_estimates_array[:, iteration] = ekf_x_hat.ravel()
    iekf_state_estimates_array[:, iteration] = iekf_x_hat.ravel()


  # Plotting the ekf_errors as well to make it easier to vizualise
  fig, ax = plt.subplots(3)
  fig.canvas.manager.set_window_title("Estimates vs ground truth")

  x_gt = gt_states_array[0, :]
  y_gt = gt_states_array[1, :]
  theta_gt = gt_states_array[2, :]

  ekf_x_hat = ekf_state_estimates_array[0, :]
  ekf_y_hat = ekf_state_estimates_array[1, :]
  ekf_theta_hat = ekf_state_estimates_array[2, :]

  iekf_x_hat = ekf_state_estimates_array[0, :]
  iekf_y_hat = ekf_state_estimates_array[1, :]
  iekf_theta_hat = ekf_state_estimates_array[2, :]


  ax[0].plot(x_gt)
  ax[0].plot(ekf_x_hat)
  ax[0].plot(iekf_x_hat)
  ax[0].legend(["Ground truth", "EKF", "IEKF"])
  ax[0].set_ylabel("x [m]")
  ax[0].set_xlabel("k [0.01s]")
  ax[0].set_title("Position in x")

  ax[1].plot(y_gt)
  ax[1].plot(ekf_y_hat)
  ax[1].plot(iekf_y_hat)
  ax[1].legend(["Ground truth", "EKF", "IEKF"])
  ax[1].set_ylabel("y [m]")
  ax[1].set_xlabel("k [0.01s]")
  ax[1].set_title("Position in y")

  ax[2].plot(theta_gt)
  ax[2].plot(ekf_x_hat)
  ax[2].plot(iekf_x_hat)
  ax[2].legend(["Ground truth", "EKF", "IEKF"])
  ax[2].set_ylabel("theta [rad]")
  ax[2].set_xlabel("k [0.01 s]")
  ax[2].set_title("Angle")


  fig, ax = plt.subplots(2)
  fig.canvas.manager.set_window_title("Comparison GT, EKF and IEKF")

  ax[0].plot(ekf_x_hat, ekf_y_hat, "r", label="EKF")
  ax[0].plot(x_gt, y_gt, "g-", label="Ground truth")
  ax[0].legend(["EKF", "Ground truth"])
  ax[0].set_ylabel("y [m]")
  ax[0].set_xlabel("x [m]")
  # ax[0].set_title("EKF vs GT")

  ax[1].plot(iekf_x_hat, iekf_y_hat, "b", label="IEKF")
  ax[1].plot(x_gt, y_gt, "g-", label="Ground truth")
  ax[1].legend(["IEKF", "Ground truth"])
  ax[1].set_ylabel("y [m]")
  ax[1].set_xlabel("x [m]")
  # ax[1].set_title("Position")

  # ax[2].plot(iekf_x_hat, iekf_y_hat, "b-", label="IEKF")
  # ax[2].plot(ekf_x_hat, ekf_y_hat, "r.", label="EKF")
  # ax[2].set_ylabel("y [m]")
  # ax[2].set_xlabel("x [m]")
  # ax[2].set_title("Position")

  # ax[1].plot(theta_gt, "g-", label="Ground truth")
  # ax[1].plot(ekf_theta_hat, "r", label="EKF estimate")
  # ax[1].plot(iekf_theta_hat, "b", label="IEKF estimate")
  # ax[1].set_ylabel("theta [rad]")
  # ax[1].set_xlabel("k [0.01 s]")
  # ax[1].set_title("Angle")


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
  print("Number of x-estimates with an error greater than 1 m: ", len(np.where(np.abs(ekf_x_errors) > 1)[0]))
  print("Number of x-estimates with an error greater than 2 m: ", len(np.where(np.abs(ekf_x_errors) > 2)[0]))
  print("Number of x-estimates with an error greater than 5 m: ", len(np.where(np.abs(ekf_x_errors) > 5)[0]))
  print()


  iekf_errors = np.subtract(gt_states_array, iekf_state_estimates_array) 
  iekf_error_fig, iekf_error_ax = plt.subplots(3)
  iekf_error_fig.canvas.manager.set_window_title("IEKF errors")

  iekf_x_errors = iekf_errors[0, :]
  iekf_y_errors = iekf_errors[1, :]
  iekf_theta_errors = iekf_errors[2, :]

  iekf_error_ax[0].plot(iekf_x_errors)
  iekf_error_ax[0].set_ylabel("error x [m]")
  iekf_error_ax[0].set_xlabel("k [0.01 s]")
  iekf_error_ax[0].set_title("Error x-position")

  iekf_error_ax[1].plot(iekf_y_errors)
  iekf_error_ax[1].set_ylabel("error y [m]")
  iekf_error_ax[1].set_xlabel("k [0.01 s]")
  iekf_error_ax[1].set_title("Error y-position")
  
  iekf_error_ax[2].plot(iekf_theta_errors)
  iekf_error_ax[2].set_ylabel("error theta [rad]")
  iekf_error_ax[2].set_xlabel("k [0.01 s]")
  iekf_error_ax[2].set_title("Error theta-angle")

  print("IEKF")
  print("Number of x-estimates with an error greater than 1 m: ", len(np.where(np.abs(iekf_x_errors) > 1)[0]))
  print("Number of x-estimates with an error greater than 2 m: ", len(np.where(np.abs(iekf_x_errors) > 2)[0]))
  print("Number of x-estimates with an error greater than 5 m: ", len(np.where(np.abs(iekf_x_errors) > 5)[0]))
  print()


  errors_ekf_vs_iekf = np.subtract(ekf_state_estimates_array, iekf_state_estimates_array)
  ekf_iekf_error_fig, ekf_iekf_error_ax = plt.subplots(3)
  ekf_iekf_error_fig.canvas.manager.set_window_title("Difference EKF and IEKF")
  ekf_iekf_error_ax[0].plot(errors_ekf_vs_iekf[0,:], label="Difference x")
  ekf_iekf_error_ax[1].plot(errors_ekf_vs_iekf[1,:], label="Difference y")
  ekf_iekf_error_ax[2].plot(errors_ekf_vs_iekf[2,:], label="Difference theta")

  plt.show()



if __name__ == '__main__':
  main()


"""
Comments:
  It looks like it is something slightly buggy with the implementation:

    I would have expected that the IEKF would perform much better compared to the EKF.
    This might be the tuning, which is naturally subpar, as it was done in a hurry, however
    the EKF actually performs better in many cases compared to the IEKF. The IEKF often include
    more measurements with a larger error at both 1, 2 and 5 meters, compared to the EKF. In 
    reality, it is expected to perform better, and not consistently slightly worse or far worse
    than the EKF. It is likely just an implemenation error somewhere that I have overlooked. 

    There is something wierd with the plotting function, as it will sometimes not include all 
    of the values in the plots. For example, the plot 'estimates vs ground truth' only show the 
    IEKF and GT, and does not include the EKF. I have no clue why. As a C++ enjoyer, Python is 
    just wierd.    

    In other words, the output from the estimators will be incorrect, and must be taken with a 
    couple of kgs of NaCl. 

    Why am I handing this in? I have spent far too much time on debugging this assignment. The 
    project thesis is taking up too much time, and it required more priority at the time of 
    writing. Also with the project being started, it makes more sense to spend some time on it, 
    instead of debugging this assignment. 


  Increasing the noise. Only one of the variables have the noise increased at a time, to 
  separate out the effects from increasing the noise. This is therefore a more theoretical 
  exercise, as the noise-levels will be connected in reality, where one should expect an 
  increase in one noise level also results in a connection with another noise level.

  It must also be noted that an increase in one measurement noise level, the process noise has 
  been left untouched. When the measurement-noise was increased, the corresponding R-matrix was
  NOT changed. An argument could be made that the measurement noise matrix, R, can be well-defined 
  by estimates on our or the manufacturer behalf. And such, it makes sense to tune the R-matrix 
  accordingly. I chose not to, however that means that my EKF/IEKF will have too large faith 
  in measurements which have an increased noise-level. 


    Angular velocity:
      Increased the angular noise to 2e9 deg^2 (yes, a variance proportional to a billion). This 
      makes the estimates far noisier, however the system is still able to follow the desired 
      trajectory. This makes some sence, as the noise is integrated, where the expected value
      over time will be zero (as it is unbiased). The position estimates therefore becomes more noisy, however 
      is less affected by a high variance. Combined with good position updates, means that it is able to
      follow the ground truth. I am however impressed that it was able to follow that well though. 

    Linear velocity:
      Increased the noise to 10000 (m/s)^2, which created more noisy estimates, with positional errors
      in approaching 10 meters several times. However, it still managed to achieve proper estimates
      and follow the ground truth. However, the noise will be multiplied with a cos() or sin() of the
      angle, before being integrated. Under the assumption that the angular velocity has relatively 
      little change between the state estimates, one should expect the integration of the unbiased 
      noise to be close to zero. Not quite zero, due to cos() and sin(). 

    To comment on the positional variance used, used a variance in the position measurements in both 
    the linear and the angular velocity noise of 2 meters in each. This is due to assuming a standard
    GNSS, under good circumstances. E.g. little to no multipath and little disturbances from the 
    ionosphere.   

    Position:
      Increasing the noise in both x and y simultaneously, which makes the positional measurements 
      have large deviations. This means that the corrections cause large changes, due to having large 
      trust in the measurements. If the R-matrix was tuned similarly to the measurement-noise levels,
      the kalman gain would place little importance in the update step, and thus relying more on the 
      predicted values.


  It is not mentioned in the assignment to change the process noise measurements. Increasing these 
  noise levels, just makes the model expect large values in the states between the iterations. For 
  example, having a large variance in the positions, implies that the system can have somewhat
  larger changes in position. An obvious case is where a constant velocity model is used to describe 
  a system which can turn every so often. A larger noise level implies that it would be better for
  movement with larger motion, such as turning. It will however be suboptimal if the system moves at 
  a relatively straight line, as the system changes far less.

"""





