import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are, inv
from numpy.linalg import matrix_rank
import time as timer

# Start the timer
start_time = timer.time()

# Motor Parameters
J = 0.01      # Inertia (kg*m^2)
b = 0.1       # Friction coefficient (N*m*s)
K_t = 1       # Motor torque constant (N*m/A)
K_e = 0.01    # Back EMF constant (V*s/rad)
R_a = 1.0     # Armature resistance (Ohm)
L_a = 0.001   # Armature inductance (H)

# Simulation Parameters
t_start = 0.0
t_end = 0.05
dt = 0.00001  # Smaller time step for Euler integration
time = np.arange(t_start, t_end, dt)
num_steps = len(time)

# Initial Conditions for the System [omega, I_a]
x_init = np.array([0.0, 0.0])  # True system state [omega, I_a]
motor_model = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)

# Desired Eigenvalues for Observer
lambda_1 = -12
lambda_2 = -10

# Initialize the Observer
x_hat_init = np.array([0.0, 0.0])  # Initial guess for the observer state [omega_hat, I_a_hat]
observer = Observer(motor_model.A, motor_model.B, motor_model.C, dt, x_hat_init)
observer.ComputeObserverGains(lambda_1, lambda_2)


# initializing MPC
# Define the matrices
num_states = 2
num_controls = 1
constraints_flag = False
# ATTENTION! here we do not use the MPC but we only use its function to compute A,B,Q and R 
# Horizon length
N_mpc = 10
# Initialize the regulator model
regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states,constr_flag=constraints_flag)
# define system matrices
regulator.setSystemMatrices(dt,motor_model.getA(),motor_model.getB())
# check the stability of the discretized system
regulator.checkStability()
# check controlability of the discretized system
regulator.checkControllabilityDiscrete()
# Define the cost matrices
Qcoeff = [200,0.0]
Rcoeff = [0.01]*num_controls

regulator.setCostMatrices(Qcoeff,Rcoeff)

Q,R = regulator.getCostMatrices()

# System Matrices
A = regulator.getDiscreteA()
B = regulator.getDiscreteB()
C = np.array([[1, 0]])  # Output matrix to measure omega only

# Augmented System Matrices
A_aug = np.block([
    [A, np.zeros((2, 1))],
    [C, np.zeros((1, 1))]
])
B_aug = np.vstack([B, np.zeros((1, 1))])
Q_aug = np.block([
    [np.diag([200, 0]), np.zeros((2, 1))],
    [np.zeros((1, 2)), 1e4]
])
R_aug = np.array([[0.01]])

# Solve DARE for the augmented system
P_aug = solve_discrete_are(A_aug, B_aug, Q_aug, R_aug)
K_aug = inv(R_aug + B_aug.T @ P_aug @ B_aug) @ B_aug.T @ P_aug @ A_aug

# Extract Gains
K_x = K_aug[:, :2]  # Gains for the original state
K_i = K_aug[:, 2:]  # Gain for the integral state

# Desired State
x_ref = np.array([10, 0])  # Reference state [omega, I_a]
x_ref_aug = np.hstack([x_ref, 0])  # Augmented reference state

# Preallocate arrays for results
omega = np.zeros(num_steps)
I_a = np.zeros(num_steps)
x_i = np.zeros(num_steps)
control_input = np.zeros(num_steps)

# Initialize states
x_cur = x_init
x_cur_aug = np.hstack([x_cur, 0])  # Augmented state

# Simulation loop for augment system
for k in range(num_steps):
    # Compute the control input
    error = x_cur_aug - x_ref_aug
    u = -K_x @ error[:2] - K_i @ error[2:]
    control_input[k] = u

    # Update integral state (last state in x_cur_aug)
    x_cur_aug[-1] += error[0] * dt  # Integrate the error in omega

    # Step the system
    motor_model.step(u)
    x_cur = motor_model.getCurrentState()
    x_cur_aug[:2] = x_cur  # Update the non-integral states

    # Store results
    omega[k] = x_cur[0]
    I_a[k] = x_cur[1]
    x_i[k] = x_cur_aug[-1]

# Stop the timer
end_time = timer.time()
elapsed_time = end_time - start_time
print(f"Time taken to run the simulation: {elapsed_time:.4f} seconds")

# Calculate Metrics
settling_time_threshold = 0.02 * np.abs(x_ref[0])
settling_indices = np.where(np.abs(omega - x_ref[0]) > settling_time_threshold)[0]
settling_time = time[settling_indices[-1]] if len(settling_indices) > 0 else 0
overshoot = (np.max(omega) - x_ref[0]) / x_ref[0] * 100
steady_state_error = np.abs(omega[-1] - x_ref[0])

# Print Metrics
print(f"Settling Time: {settling_time:.4f} s")
print(f"Overshoot: {overshoot:.2f} %")
print(f"Steady-State Error: {steady_state_error:.2f}")

# Plot Results
plt.figure(figsize=(12, 10))

# Angular velocity
plt.subplot(4, 1, 1)
plt.plot(time, omega, label='True $\omega$ (rad/s)')
plt.axhline(x_ref[0], color='r', linestyle='--', label='Reference $\omega$')
plt.title('Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid()

# Armature current
plt.subplot(4, 1, 2)
plt.plot(time, I_a, label='True $I_a$ (A)')
plt.title('Armature Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid()

# Integral of output error
plt.subplot(4, 1, 3)
plt.plot(time, x_i, label='Integral of Error ($x_i$)')
plt.title('Integral of Output Error')
plt.xlabel('Time (s)')
plt.ylabel('Integral of Error')
plt.legend()
plt.grid()

# Control input
plt.subplot(4, 1, 4)
plt.plot(time, control_input, label='Control Input ($u$)')
plt.title('Control Input')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (V)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
