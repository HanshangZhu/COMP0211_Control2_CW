import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
from regulator_model import RegulatorModel
from scipy.linalg import solve_discrete_are, inv

# Nominal Motor Parameters
J = 0.01
b = 0.1
K_t = 1
K_e_nominal = 0.01  # Back EMF constant
R_a_nominal = 1.0   # Armature resistance
L_a_nominal = 0.001 # Armature inductance

# Perturbation Range (±20%)
perturbation_range = 0.2
parameter_sets = [
    {
        "K_e": K_e_nominal * (1 + factor),
        "R_a": R_a_nominal * (1 + factor),
        "L_a": L_a_nominal * (1 + factor),
    }
    for factor in [-perturbation_range, 0, perturbation_range]
]

# Simulation Parameters
t_start, t_end, dt = 0.0, 0.01, 0.00001
time = np.arange(t_start, t_end, dt)
num_steps = len(time)

# Initialize results storage
results = {"MPC": [], "LQR": []}


# Loop through each parameter set
for params in parameter_sets:
    print(f"Testing with parameters: {params}")
    K_e, R_a, L_a = params["K_e"], params["R_a"], params["L_a"]

    # Initialize System
    x_init = np.array([0.0, 0.0])
    motor_model = SysDyn(J, b, K_t, K_e, R_a, L_a, dt, x_init)

    # Observer
    lambda_1, lambda_2 = -12, -10
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

        if k == num_steps - 1:  # After the simulation loop is complete for this parameter set
            results["MPC"].append(omega_mpc)
            results["LQR"].append(omega_lqr)




# Plot Results
for i, params in enumerate(parameter_sets):
    plt.figure(figsize=(12, 10))

    # Angular velocity
    plt.subplot(4, 1, 1)
    plt.plot(time, results["MPC"][i], label="MPC $\omega$ (rad/s)")
    plt.plot(time, results["LQR"][i], '--', label="LQR $\omega$ (rad/s)")
    plt.title(f"Angular Velocity with Perturbed Parameters: {params}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.legend()
    plt.grid()

    # Armature current
    plt.subplot(4, 1, 2)
    plt.plot(time, I_a_mpc, label="MPC $I_a$ (A)")
    plt.plot(time, I_a_lqr, '--', label="LQR $I_a$ (A)")
    plt.title("Armature Current")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")
    plt.legend()
    plt.grid()

    # Torque
    plt.subplot(4, 1, 3)
    plt.plot(time, T_m_true_mpc, label="MPC $T_m$ (N*m)")
    plt.plot(time, T_m_true_lqr, '--', label="LQR $T_m$ (N*m)")
    plt.title("Motor Torque")
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (N*m)")
    plt.legend()
    plt.grid()

    # Terminal Voltage
    plt.subplot(4, 1, 4)
    plt.plot(time, V_terminal_mpc, label="MPC $V_{terminal}$ (V)")
    plt.plot(time, V_terminal_lqr, '--', label="LQR $V_{terminal}$ (V)")
    plt.title("Terminal Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# Metrics for MPC and LQR
metrics = {"MPC": {"settling_time": [], "overshoot": [], "steady_state_error": [], "control_effort": []},
           "LQR": {"settling_time": [], "overshoot": [], "steady_state_error": [], "control_effort": []}}

# Helper function to calculate settling time
def calculate_settling_time(time, response, reference, threshold=0.02):
    settling_indices = np.where(np.abs(response - reference) > threshold * reference)[0]
    if len(settling_indices) > 0:
        return time[settling_indices[-1]]
    return 0

# Compute metrics for each parameter set
for i in range(len(parameter_sets)):
    # Settling Time
    metrics["MPC"]["settling_time"].append(calculate_settling_time(time, results["MPC"][i], x_ref[0]))
    metrics["LQR"]["settling_time"].append(calculate_settling_time(time, results["LQR"][i], x_ref[0]))

    # Overshoot
    metrics["MPC"]["overshoot"].append((np.max(results["MPC"][i]) - x_ref[0]) / x_ref[0] * 100)
    metrics["LQR"]["overshoot"].append((np.max(results["LQR"][i]) - x_ref[0]) / x_ref[0] * 100)

    # Steady-State Error
    metrics["MPC"]["steady_state_error"].append(np.abs(results["MPC"][i][-1] - x_ref[0]))
    metrics["LQR"]["steady_state_error"].append(np.abs(results["LQR"][i][-1] - x_ref[0]))

    # Control Effort (e.g., sum of |u| over time)
    metrics["MPC"]["control_effort"].append(np.sum(np.abs(V_terminal_mpc)))
    metrics["LQR"]["control_effort"].append(np.sum(np.abs(V_terminal_lqr)))

# Display Metrics
for controller in ["MPC", "LQR"]:
    print(f"\n{controller} Metrics:")
    for metric, values in metrics[controller].items():
        print(f"{metric}: {values}")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
parameter_labels = [f"{params}" for params in parameter_sets]

# Settling Time
axs[0, 0].bar(parameter_labels, metrics["MPC"]["settling_time"], label="MPC")
axs[0, 0].bar(parameter_labels, metrics["LQR"]["settling_time"], label="LQR", alpha=0.7)
axs[0, 0].set_title("Settling Time")
axs[0, 0].set_ylabel("Time (s)")
axs[0, 0].legend()

# Overshoot
axs[0, 1].bar(parameter_labels, metrics["MPC"]["overshoot"], label="MPC")
axs[0, 1].bar(parameter_labels, metrics["LQR"]["overshoot"], label="LQR", alpha=0.7)
axs[0, 1].set_title("Overshoot")
axs[0, 1].set_ylabel("Percentage (%)")
axs[0, 1].legend()

# Steady-State Error
axs[1, 0].bar(parameter_labels, metrics["MPC"]["steady_state_error"], label="MPC")
axs[1, 0].bar(parameter_labels, metrics["LQR"]["steady_state_error"], label="LQR", alpha=0.7)
axs[1, 0].set_title("Steady-State Error")
axs[1, 0].set_ylabel("Error (rad/s)")
axs[1, 0].legend()

# Control Effort
axs[1, 1].bar(parameter_labels, metrics["MPC"]["control_effort"], label="MPC")
axs[1, 1].bar(parameter_labels, metrics["LQR"]["control_effort"], label="LQR", alpha=0.7)
axs[1, 1].set_title("Control Effort")
axs[1, 1].set_ylabel("Sum of |u(t)|")
axs[1, 1].legend()

plt.tight_layout()
plt.show()
