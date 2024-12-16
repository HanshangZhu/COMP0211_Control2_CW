import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are, inv
import pandas as pd

# Nominal Motor Parameters
J = 0.01
b = 0.1
K_t = 1
K_e_nominal = 0.01  # Back EMF constant
R_a_nominal = 1.0   # Armature resistance
L_a_nominal = 0.001 # Armature inductance

# Perturbation Range (±20%)
perturbation_range = 0.2
num_iterations = 10

parameter_sets = [
    {
        "K_e": K_e_nominal * (1 + np.random.uniform(-perturbation_range, perturbation_range)),
        "R_a": R_a_nominal * (1 + np.random.uniform(-perturbation_range, perturbation_range)),
        "L_a": L_a_nominal * (1 + np.random.uniform(-perturbation_range, perturbation_range)),
    }
    for _ in range(num_iterations)
]

# Simulation Parameters
t_start, t_end, dt = 0.0, 0.01, 0.00001
time = np.arange(t_start, t_end, dt)
num_steps = len(time)

# Initialize DataFrame to store results
results_table = []

# Helper function to calculate settling time
def calculate_settling_time(time, response, reference, threshold=0.02):
    settling_indices = np.where(np.abs(response - reference) > threshold * reference)[0]
    if len(settling_indices) > 0:
        return time[settling_indices[-1]]
    return 0

# Loop through each parameter set
for params in parameter_sets:
    print(f"Testing with parameters: {params}")
    K_e, R_a, L_a = params["K_e"], params["R_a"], params["L_a"]

    # Initialize System
    x_init = np.array([0.0, 0.0])
    motor_model = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)

    # Observer
    lambda_1, lambda_2 = -20, -10
    observer = Observer(motor_model.A, motor_model.B, motor_model.C, dt, x_init)
    observer.ComputeObserverGains(lambda_1, lambda_2)
    
    # MPC Controller Initialisation
    # initializing MPC (regulator model)
    # Define the matrices
    num_states = 2
    num_controls = 1
    constraints_flag = False

    # Horizon length
    N_mpc = 10

    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states,constr_flag=constraints_flag)

    # define system matrices (discretise matrix A and B based on delta_t)
    regulator.setSystemMatrices(dt,motor_model.getA(),motor_model.getB())

    # check the stability of the discretized system
    regulator.checkStability()

    # check controlability of the discretized system
    regulator.checkControllabilityDiscrete()

    # Define the cost matrices
    Qcoeff = [107.0,0.0]
    Rcoeff = [0.01]*num_controls

    # Making up the Q and R matrice with our defined Q,Rcoeff in the matrices' diagonals.
    regulator.setCostMatrices(Qcoeff,Rcoeff)

    # Desired State
    x_ref = np.array([10,0])

    # Setting up Constraints
    regulator.propagation_model_regulator_fixed_std(x_ref)

    B_in = {'max': np.array([100000000000000] * num_controls), 'min': np.array([-1000000000000] * num_controls)}
    B_out = {'max': np.array([100000000,1000000000]), 'min': np.array([-100000000,-1000000000])}
    # creating constraints matrices
    regulator.setConstraintsMatrices(B_in,B_out)
    regulator.compute_H_and_F()

    # LQR Controller
    Q,R = regulator.getCostMatrices()
    A = regulator.getDiscreteA()
    B = regulator.getDiscreteB()
    P = solve_discrete_are(A, B, Q, R)
    K_lqr = inv(R + B.T @ P @ B) @ B.T @ P @ A
    B_pinv = np.linalg.pinv(B)  # Result is a (1x2) matrix
    # Compute Delta x (this is for the discrete system)
    delta_x = A @ x_ref # Result is a (2x1) vector
    # Compute u_ff
    u_ff = - B_pinv @ delta_x 

    # Preallocate arrays for storing results for MPC
    omega_mpc = np.zeros(num_steps)
    I_a_mpc = np.zeros(num_steps)
    hat_omega_mpc = np.zeros(num_steps)
    hat_I_a_mpc = np.zeros(num_steps)
    T_m_true_mpc = np.zeros(num_steps)
    T_m_estimated_mpc = np.zeros(num_steps)
    V_terminal_mpc = np.zeros(num_steps)
    V_terminal_hat_mpc = np.zeros(num_steps)

    # Preallocate arrays for storing results for LQR
    omega_lqr = np.zeros(num_steps)
    I_a_lqr = np.zeros(num_steps)
    hat_omega_lqr = np.zeros(num_steps)
    hat_I_a_lqr = np.zeros(num_steps)
    T_m_true_lqr = np.zeros(num_steps)
    T_m_estimated_lqr = np.zeros(num_steps)
    V_terminal_lqr = np.zeros(num_steps)
    V_terminal_hat_lqr = np.zeros(num_steps)

    # Initialize separate states for MPC and LQR
    x_cur_mpc = x_init.copy()
    x_cur_lqr = x_init.copy()
    x_hat_cur_mpc = x_init.copy()
    x_hat_cur_lqr = x_init.copy()

    for k in range(num_steps):
        # Time stamp
        t = time[k]

        # Disturbance: Add external torque disturbance
        disturbance = 0.1 * np.sin(2 * np.pi * 5 * time[k])  # Example disturbance

        # LQR Control
        V_lqr = -K_lqr @ (x_cur_lqr - x_ref) + u_ff
        y_cur_lqr = motor_model.step(V_lqr + disturbance)
        x_cur_lqr = motor_model.getCurrentState()
        V_terminal_lqr[k] = y_cur_lqr
        x_hat_cur_lqr,y_hat_cur_lqr = observer.update(V_lqr + disturbance, y_cur_lqr)

        # Store results for LQR
        omega_lqr[k] = x_cur_lqr[0]
        I_a_lqr[k] = x_cur_lqr[1]
        hat_omega_lqr[k] = x_hat_cur_lqr[0]
        hat_I_a_lqr[k] = x_hat_cur_lqr[1]
        T_m_true_lqr[k] = K_t * I_a_lqr[k]
        T_m_estimated_lqr[k] = K_t * hat_I_a_lqr[k]
        V_terminal_hat_lqr[k] = y_hat_cur_lqr


        # MPC Control
        u_mpc = regulator.compute_solution(x_hat_cur_mpc)
        V_mpc = u_mpc[0]
        y_cur_mpc = motor_model.step(V_mpc + disturbance)
        x_cur_mpc = motor_model.getCurrentState()
        V_terminal_mpc[k] = y_cur_mpc
        x_hat_cur_mpc,y_hat_cur_mpc = observer.update(V_mpc + disturbance, y_cur_mpc)

        # Store results for mpc
        omega_mpc[k] = x_cur_mpc[0]
        I_a_mpc[k] = x_cur_mpc[1]
        hat_omega_mpc[k] = x_hat_cur_mpc[0]
        hat_I_a_mpc[k] = x_hat_cur_mpc[1]
        T_m_true_mpc[k] = K_t * I_a_mpc[k]
        T_m_estimated_mpc[k] = K_t * hat_I_a_mpc[k]
        V_terminal_hat_mpc[k] = y_hat_cur_mpc

    # Calculate Metrics after Simulation
    mpc_settling_time = calculate_settling_time(time, omega_mpc, x_ref[0])
    mpc_overshoot = (np.max(omega_mpc) - x_ref[0]) / x_ref[0] * 100
    mpc_steady_state_error = np.abs(omega_mpc[-1] - x_ref[0])

    lqr_settling_time = calculate_settling_time(time, omega_lqr, x_ref[0])
    lqr_overshoot = (np.max(omega_lqr) - x_ref[0]) / x_ref[0] * 100
    lqr_steady_state_error = np.abs(omega_lqr[-1] - x_ref[0])

    # Append Results
    results_table.append({
        "K_e": K_e, "R_a": R_a, "L_a": L_a,
        "MPC Settling Time": mpc_settling_time, "MPC Overshoot (%)": mpc_overshoot,
        "MPC Steady-State Error": mpc_steady_state_error,
        "LQR Settling Time": lqr_settling_time, "LQR Overshoot (%)": lqr_overshoot,
        "LQR Steady-State Error": lqr_steady_state_error
    })

from tabulate import tabulate

# Convert Results to DataFrame and Display
df = pd.DataFrame(results_table)

# Calculate correlation between parameters and performance metrics
correlation_matrix = df.corr()

# Convert DataFrame to a pretty table
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
print(tabulate(correlation_matrix, headers='keys', tablefmt='fancy_grid', showindex=False))


import matplotlib.pyplot as plt

# List of relationships to plot
relationships = [
    ("K_e", "MPC Settling Time"), ("K_e", "MPC Overshoot (%)"), ("K_e", "MPC Steady-State Error"),
    ("R_a", "MPC Settling Time"), ("R_a", "MPC Overshoot (%)"), ("R_a", "MPC Steady-State Error"),
    ("L_a", "MPC Settling Time"), ("L_a", "MPC Overshoot (%)"), ("L_a", "MPC Steady-State Error"),
    ("K_e", "LQR Settling Time"), ("K_e", "LQR Overshoot (%)"), ("K_e", "LQR Steady-State Error"),
    ("R_a", "LQR Settling Time"), ("R_a", "LQR Overshoot (%)"), ("R_a", "LQR Steady-State Error"),
    ("L_a", "LQR Settling Time"), ("L_a", "LQR Overshoot (%)"), ("L_a", "LQR Steady-State Error"),
]

# Create scatter plots
fig, axes = plt.subplots(6, 3, figsize=(15, 18))
fig.suptitle("Relationships Between System Parameters and Performance Metrics", fontsize=16)

# Iterate over relationships and plot
for ax, (x_col, y_col) in zip(axes.flat, relationships):
    ax.scatter(df[x_col], df[y_col], alpha=0.7, edgecolors='k')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    ax.grid(True)

# Adjust layout and show
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
