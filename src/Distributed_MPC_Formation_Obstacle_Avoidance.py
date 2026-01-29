import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

# ==============================================================================
# === 1. Parameters ============================================================
# ==============================================================================

N_agents = 6
n = 2  # State dimension (x, y)
n_z = 2 * n  # Full state dimension [x, y, vx, vy]
n_u = n      # Input dimension

T = 25  # Simulation time steps
N = 15  # Prediction horizon
R = 0.25  # Safety radius

# Cost function weights
alpha_i = 1.0
beta_i = 0.1
rho_i = 0.1
gamma_i = 0.9  # Parameter for compatibility constraint
Qi_neg_def = -0.01 * np.eye(n_z) # Negative definite matrix for Lyapunov eqn

# System matrices (Discrete-time second-order agent)
In = np.eye(n)
O = np.zeros((n,n))

A_np = np.block([[In, In],
                 [O,  In]])
B_np = np.block([[0.5 * In],
                 [In]])

# Input constraints |u_i,j| <= 0.5
u_min = np.array([-0.5, -0.5]).reshape(-1,1)
u_max = np.array([0.5, 0.5]).reshape(-1,1)
u_bounds = (u_min, u_max)

# State constraints (chosen to be sufficiently large)
z_min = np.array([-15, -15, -5, -5]).reshape(-1,1)
z_max = np.array([15, 15, 5, 5]).reshape(-1,1)
Z_bounds = (z_min, z_max)


# Initial states zi(0) = [x, y, vx, vy]
z_init = np.array([
    [-3, -3, -0.5,  0.5],
    [-5, -2,  0.5,  0.5],
    [-4, -7,  0.0,  0.5],
    [-2, -5, -0.5,  0.5],
    [-6, -4, -0.5, -0.5],
    [-2, -6,  0.0,  0.0]
])

# Desired positions x_i^d for each agent
xd = np.array([
    [0,  0],
    [1,  0],
    [2,  1],
    [1,  2],
    [0,  2],
    [-1, 1]
])

# Desired states z_i^d = [x_i^d.T, 0].T
zd = np.hstack([xd, np.zeros_like(xd)])

# Obstacle position
x_obs = np.array([[-2, -2]])

# Adjacency matrix defining neighbors
adj_matrix = np.array([
    [0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0]
])
neighbors_list = [np.where(row == 1)[0] for row in adj_matrix]

# Desired relative positions d_ij = x_j^d - x_i^d
dij_map = {}
for i in range(N_agents):
    for j in neighbors_list[i]:
        dij_map[(i, j)] = xd[j] - xd[i]

# Convert d_ij to d_ij^z = [d_ij.T, 0]^T
dij_z_map = {k: np.hstack([v, np.zeros(n)]) for k, v in dij_map.items()}

# ==============================================================================
# === 2. Offline Stage: Compute Terminal Ingredients ===========================
# ==============================================================================

def compute_terminal_controller(A, B):
    """Computes K1 using LQR to stabilize the system."""
    Q_lqr = np.eye(n_z)
    R_lqr = np.eye(n_u)
    P_are = solve_discrete_are(A, B, Q_lqr, R_lqr)
    K1 = -np.linalg.inv(R_lqr + B.T @ P_are @ B) @ (B.T @ P_are @ A)
    return K1

def compute_K2(K1):
    """Computes K2 such that L1 = (K1+K2)[:n, :n] = 0"""
    K2 = np.zeros_like(K1)
    K2[:, :n] = -K1[:, :n]
    return K2

def compute_Pi(A, B, K1, alpha, beta, rho, neighbors_i, Qi_neg_def):
    """Solves the Lyapunov equation (30) to find Pi."""
    Phi = A + B @ K1
    sum_beta_term = sum(2 * (beta + beta) for j in neighbors_i) * np.eye(n_z) # beta_i = beta_j
    Q_tilde = Qi_neg_def - alpha * np.eye(n_z) - rho * K1.T @ K1 - sum_beta_term
    Pi = solve_discrete_lyapunov(Phi.T, -Q_tilde)
    return Pi

def compute_Di(xd, dij_map, x_obs, neighbors_list, R):
    """Computes the terminal set size Di."""
    Di_list = []
    for i in range(N_agents):
        D_i1_vals = [(np.linalg.norm(dij_map[(i, j)]) - 2 * R) / 2 for j in neighbors_list[i]]
        D_i1 = min(D_i1_vals) if D_i1_vals else np.inf
        D_i2 = min([np.linalg.norm(xd[i] - obs) for obs in x_obs]) - 2 * R
        Di_list.append(min(D_i1, D_i2))
    return np.array(Di_list)

# --- Execute Offline Computations ---
K1 = compute_terminal_controller(A_np, B_np)
K2 = compute_K2(K1)
P_all = [compute_Pi(A_np, B_np, K1, alpha_i, beta_i, rho_i, neighbors, Qi_neg_def) for neighbors in neighbors_list]
D_all = compute_Di(xd, dij_map, x_obs, neighbors_list, R)

# ==============================================================================
# === 3. DMPC Solver: Formulation of Problem 2 ================================
# ==============================================================================

def solve_dmpc_problem(z_current, z_desired, agent_idx, z_hat_self, z_hat_neighbors, k, A, B, u_initial_guess):
    """
    Solves the DMPC optimization problem (Problem 2) for a single agent.
    """
    opti = ca.Opti()

    # --- Decision variables ---
    U = opti.variable(N, n_u)

    # slack_pos = opti.variable()
    # slack_state = opti.variable()

    # --- System dynamics ---
    Z = [ca.MX(z_current)]
    for l in range(N):
        Z.append(A @ Z[l] + B @ U[l, :].T)

    # --- Cost function (Corrected version) ---
    cost = 0

    # cost += 1e7 * (slack_pos**2 + slack_state**2)
    # opti.subject_to(slack_pos >= 0)
    # opti.subject_to(slack_state >= 0)

    # Stage cost L_i for l = 0...N-1
    for l in range(N):
        z_l = Z[l]
        u_l = U[l, :]
        cost += alpha_i * ca.sumsqr(z_desired - z_l)
        for j_local, neighbor_global_idx in enumerate(neighbors_list[agent_idx]):
            z_hat_j = z_hat_neighbors[j_local][l, :]
            d_ij_z = dij_z_map[(agent_idx, neighbor_global_idx)]
            cost += beta_i * ca.sumsqr(z_hat_j - z_l - d_ij_z)
        cost += rho_i * ca.sumsqr(u_l)

    # Terminal cost L_if for Z[N]
    z_N = Z[N]
    P_i = P_all[agent_idx]
    cost += ca.mtimes([(z_N - z_desired).T, P_i, (z_N - z_desired)])
    opti.minimize(cost)

    # --- Constraints ---
    x_hat_self_pos = z_hat_self[:, :n]

    # 1. Input constraints: l = 0 to N-1
    for l in range(N):
        opti.subject_to(opti.bounded(u_min, U[l, :].T, u_max))

    # 2. State constraints: l = 1 to N
    for l in range(1, N + 1):
        z_i = Z[l]
        x_i = z_i[:n]

        opti.subject_to(opti.bounded(z_min, z_i, z_max))

        # Collision Avoidance
        for j_local, neighbor_global_idx in enumerate(neighbors_list[agent_idx]):
            x_hat_j = z_hat_neighbors[j_local][l-1, :n]
            x_hat_i = x_hat_self_pos[l-1, :]
            mu_ij = (ca.norm_2(x_hat_j - x_hat_i) - 2 * R) / 2
            opti.subject_to(ca.norm_2(x_hat_j - x_i) >= 2 * R + mu_ij)

        # Obstacle Avoidance
        for obs in x_obs:
            opti.subject_to(ca.norm_2(obs - x_i) >= 2 * R)

        # Compatibility Constraints
        # mu_ij_list = []
        # for j_local in range(len(neighbors_list[agent_idx])):
        #      x_hat_j = z_hat_neighbors[j_local][l-1, :n]
        #      x_hat_i = x_hat_self_pos[l-1, :]
        #      mu_ij_list.append((ca.norm_2(x_hat_j - x_hat_i) - 2*R)/2)
        # mu_i = ca.mmin(ca.vertcat(*mu_ij_list)) if mu_ij_list else 1e9
        # opti.subject_to(ca.norm_2(x_hat_self_pos[l-1, :] - x_i) <= mu_i)

        # a_i = sum(beta_i + 2 * beta_i for j in neighbors_list[agent_idx])
        # phi_ji_list = []
        # for j_local, neighbor_global_idx in enumerate(neighbors_list[agent_idx]):
        #     d_ij_z = dij_z_map[(agent_idx, neighbor_global_idx)]
        #     phi_ji = ca.mmax(ca.vertcat(*[ca.norm_2(z_hat_neighbors[j_local][ll, :] - z_hat_self[ll, :] - d_ij_z) for ll in range(N)]))
        #     phi_ji_list.append(phi_ji)
        # b_i = sum(2 * beta_i * phi_ji_list[j_local] for j_local in range(len(neighbors_list[agent_idx])))
        # c_i = -(gamma_i / (N - 1)) * alpha_i * ca.sumsqr(z_desired - z_current)
        # discriminant = b_i**2 - 4 * a_i * c_i

        # nu_i = (-b_i + ca.sqrt(ca.fmax(discriminant, 0) + 1e-9)) / (2 * a_i)
        # opti.subject_to(ca.norm_2(z_hat_self[l-1, :] - z_i) <= nu_i)

    # Terminal Constraint
    delta_x_N = z_desired[:n] - Z[N][:n]
    opti.subject_to(ca.norm_2(delta_x_N) <= D_all[agent_idx])

    # --- Solve ---
    s_opts = {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': 'yes', 'ipopt.max_iter': 3000}
    opti.solver('ipopt', s_opts)

    # 초기 추측치 제공
    opti.set_initial(U, u_initial_guess)

    try:
        sol = opti.solve()
        u_optimal = sol.value(U)
        z_optimal = np.array([sol.value(z) for z in Z])
        return u_optimal, z_optimal
    except Exception as e:
        print(f"Solver failed for agent {agent_idx} at time k={k}: {e}")
        return np.zeros((N, n_u)), np.tile(z_current, (N + 1, 1))

# ==============================================================================
# === 4. Main Simulation Loop (Algorithm 1) ====================================
# ==============================================================================

# --- Storage for trajectories ---
z_history = np.zeros((N_agents, T + 1, n_z))
u_history = np.zeros((N_agents, T, n_u))
# 모든 상태 벡터를 (n_z, 1) 형태의 열 벡터로 저장하기 위해 z_history 초기화 수정
for i in range(N_agents):
    z_history[i, 0, :] = z_init[i]

# --- Storage for optimal solutions at each step k ---
u_opt_k = np.zeros((N_agents, N, n_u))
z_opt_k = np.zeros((N_agents, N + 1, n_z))


# --- Initial Guess for k=0 ---
print("Generating initial guess...")
# u_opt_k는 LQR 제어 입력을, z_opt_k는 그 결과 궤적을 저장합니다.
for i in range(N_agents):
    z_current_guess = z_init[i].reshape(-1, 1)
    zd_i_col = zd[i].reshape(-1, 1)
    z_guess_traj = [z_current_guess]
    u_guess_traj = []
    for l in range(N):
        u_l = K1 @ (z_guess_traj[l] - zd_i_col)
        u_clipped = np.clip(u_l, u_min, u_max)
        u_guess_traj.append(u_clipped)
        z_next = A_np @ z_guess_traj[l] + B_np @ u_guess_traj[l]
        z_guess_traj.append(z_next)
    u_opt_k[i] = np.array(u_guess_traj).squeeze()
    z_opt_k[i] = np.array(z_guess_traj).squeeze()


# --- Main Loop ---
print("Starting DMPC simulation...")
for k in range(T):
    print(f"Time step k = {k}/{T-1}")

    # (a-2) Compute and transmit assumed trajectories
    z_hat_all = np.zeros_like(z_opt_k)
    for i in range(N_agents):
        u_hat = np.vstack([u_opt_k[i, 1:], np.zeros(n_u)])
        z_hat = np.vstack([z_opt_k[i, 1:], np.zeros(n_z)])
        terminal_state_prev_opt = z_opt_k[i, N, :]

        # zd[i]를 열 벡터로 변환하여 계산
        zd_i_col = zd[i].reshape(-1, 1)
        u_terminal = (K1 @ terminal_state_prev_opt.reshape(-1,1) + K2 @ zd_i_col).flatten()

        u_hat[N-1, :] = u_terminal
        z_hat[N, :] = A_np @ terminal_state_prev_opt + B_np @ u_hat[N-1, :]
        z_hat_all[i] = z_hat

    # (b) At time k, each agent solves its DMPC problem
    u_opt_k_new = np.zeros_like(u_opt_k)
    z_opt_k_new = np.zeros_like(z_opt_k)

    A = ca.DM(A_np)
    B = ca.DM(B_np)

    for i in range(N_agents):
        z_current_i = z_history[i, k, :]
        z_desired_i = zd[i]
        neighbors_i = neighbors_list[i]
        z_hat_self_i = z_hat_all[i]
        z_hat_neighbors_i = [z_hat_all[j] for j in neighbors_i]

        # [수정] 웜 스타트(Warm Start)를 위한 초기 추측치 준비
        if k == 0:
            # k=0일 때는 LQR 제어 입력을 초기 추측치로 사용
            u_warm_start = u_opt_k[i]
        else:
            # k>0 부터는 이전 스텝의 해를 사용 (shift)
            u_warm_start = np.vstack([u_opt_k[i, 1:], u_opt_k[i, -1]])

        u_sol, z_sol = solve_dmpc_problem(
            z_current=z_current_i,
            z_desired=z_desired_i,
            agent_idx=i,
            z_hat_self=z_hat_self_i,
            z_hat_neighbors=z_hat_neighbors_i,
            k=k, A=A, B=B,
            u_initial_guess=u_warm_start # 초기 추측치 전달
        )
        u_opt_k_new[i] = u_sol
        z_opt_k_new[i] = z_sol.squeeze() # squeeze로 차원 맞추기

    u_opt_k = u_opt_k_new
    z_opt_k = z_opt_k_new

    # (a-1) Apply first control input and update system state
    for i in range(N_agents):
        u_apply = u_opt_k[i, 0, :]
        u_history[i, k, :] = u_apply
        z_history[i, k + 1, :] = A_np @ z_history[i, k, :] + B_np @ u_apply

print("Simulation finished.")

# ==============================================================================
# === 5. Visualization =========================================================
# ==============================================================================
plt.style.use('seaborn-v0_8-whitegrid')

# --- Figure 1: Agent Trajectories ---
fig1, ax1 = plt.subplots(figsize=(8, 8))
colors = plt.cm.jet(np.linspace(0, 1, N_agents))
obstacle_patch = Circle(x_obs[0], R, color='black', label='Obstacle')
ax1.add_patch(obstacle_patch)

for i in range(N_agents):
    x_coords = z_history[i, :, 0]
    y_coords = z_history[i, :, 1]
    ax1.plot(x_coords, y_coords, '-o', color=colors[i], label=f'Agent {i+1}', markersize=3)
    ax1.plot(xd[i, 0], xd[i, 1], 'x', color=colors[i], markersize=10, mew=2) # Target

ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ax1.set_title('Trajectories of all agents')
ax1.legend()
ax1.grid(True)
ax1.set_aspect('equal', adjustable='box')
# plt.show()


# --- Figure 2 & 3: Relative Distances ---
fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
time = np.arange(T)

# Agent-to-Agent distances (Example: Agent 1 vs others)
for j in range(1, N_agents):
    dist = np.linalg.norm(z_history[0, :-1, :2] - z_history[j, :-1, :2], axis=1)
    ax2.plot(time, dist, label=f'd(1, {j+1})')
ax2.axhline(y=2*R, color='r', linestyle='--', label='Safety Distance (2R)')
ax2.set_ylabel('Distance')
ax2.set_title('Relative Distances between Agent 1 and Others')
ax2.legend()
ax2.set_ylim(bottom=0)

# Agent-to-Obstacle distances
for i in range(N_agents):
    dist = np.linalg.norm(z_history[i, :-1, :2] - x_obs[0], axis=1)
    ax3.plot(time, dist, color=colors[i], label=f'd(Agent {i+1}, Obs)')
ax3.axhline(y=2*R, color='r', linestyle='--')
ax3.set_xlabel('Time step (k)')
ax3.set_ylabel('Distance')
ax3.set_title('Relative Distances between Each Agent and Obstacle')
ax3.set_ylim(bottom=0)
plt.tight_layout()
# plt.show()

# --- Figure 4 & 5: Control Inputs ---
fig3, (ax4, ax5) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
for i in range(N_agents):
    ax4.plot(time, u_history[i, :, 0], color=colors[i], label=f'Agent {i+1}')
    ax5.plot(time, u_history[i, :, 1], color=colors[i])

ax4.axhline(y=u_max[0], color='r', linestyle='--')
ax4.axhline(y=u_min[0], color='r', linestyle='--')
ax4.set_ylabel('$u_{i,1}$')
ax4.set_title('First Component of Control Input')
ax4.legend()

ax5.axhline(y=u_max[1], color='r', linestyle='--')
ax5.axhline(y=u_min[1], color='r', linestyle='--')
ax5.set_xlabel('Time step (k)')
ax5.set_ylabel('$u_{i,2}$')
ax5.set_title('Second Component of Control Input')
plt.tight_layout()
# plt.show()

# ==============================================================================
# === 6. Animation =============================================================
# ==============================================================================
import matplotlib.animation as animation

# --- 애니메이션 설정 ---
fig_ani, ax_ani = plt.subplots(figsize=(9, 9))
ax_ani.set_xlabel('$x_1$')
ax_ani.set_ylabel('$x_2$')
ax_ani.set_title('DMPC Agent Animation')
ax_ani.grid(True)
ax_ani.set_aspect('equal', adjustable='box')

# 맵의 경계를 z_history를 기반으로 설정하여 모든 움직임을 포함하도록 함
all_x = z_history[:, :, 0].flatten()
all_y = z_history[:, :, 1].flatten()
ax_ani.set_xlim(all_x.min() - 1, all_x.max() + 1)
ax_ani.set_ylim(all_y.min() - 1, all_y.max() + 1)


# --- 정적 요소 그리기 ---
# 장애물
obstacle_patch_ani = Circle(x_obs[0], R, color='black', label='Obstacle', alpha=0.8)
ax_ani.add_patch(obstacle_patch_ani)
# 목표 지점
for i in range(N_agents):
    ax_ani.plot(xd[i, 0], xd[i, 1], 'x', color=colors[i], markersize=12, mew=2.5, label=f'Target {i+1}' if i==0 else "")

# --- 애니메이션 요소 초기화 ---
# 에이전트 원형 패치
agent_patches = [Circle(z_init[i, :2], R, color=colors[i], alpha=0.7) for i in range(N_agents)]
for patch in agent_patches:
    ax_ani.add_patch(patch)

# 에이전트 궤적 선
trail_lines = [ax_ani.plot([], [], lw=1.5, color=colors[i], label=f'Agent {i+1}')[0] for i in range(N_agents)]

# 시간 표시 텍스트
time_text = ax_ani.text(0.02, 0.95, '', transform=ax_ani.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# 범례(Legend) 추가
handles, labels = ax_ani.get_legend_handles_labels()
# Agent 레이블과 Target 레이블 순서 조정
order = [i for i in range(1, N_agents+1)] + [0] + [N_agents+1] # Agent 1~6, Obstacle, Target
ax_ani.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right')


# --- 애니메이션 업데이트 함수 ---
def update(k):
    """ 각 프레임(시간 k)마다 호출되는 함수 """
    # 각 에이전트에 대해 위치와 궤적 업데이트
    for i in range(N_agents):
        # 원의 중심 위치 업데이트
        agent_patches[i].center = (z_history[i, k, 0], z_history[i, k, 1])

        # 궤적 데이터 업데이트
        trail_lines[i].set_data(z_history[i, :k+1, 0], z_history[i, :k+1, 1])

    # 시간 텍스트 업데이트
    time_text.set_text(f'Time Step: {k}/{T}')

    # 업데이트된 객체들을 반환 (blit=True 사용 시 필수)
    return agent_patches + trail_lines + [time_text]

# --- 애니메이션 생성 및 실행 ---
# FuncAnimation 객체 생성
# interval: 프레임 간 지연 시간 (밀리초)
# blit=True: 성능 향상을 위해 변경된 부분만 다시 그림
ani = animation.FuncAnimation(
    fig=fig_ani,
    func=update,
    frames=T + 1,
    interval=100,
    blit=True,
    repeat=False
)

# 애니메이션 보여주기
# plt.show()

# --- (선택사항) 애니메이션 파일로 저장하기 ---
# 애니메이션을 저장하려면 ffmpeg와 같은 외부 프로그램이 필요할 수 있습니다.
# 아나콘다 환경의 경우: conda install -c conda-forge ffmpeg
# Ubuntu의 경우: sudo apt-get install ffmpeg
#
print("\n애니메이션을 파일로 저장하는 중... (시간이 걸릴 수 있습니다)")
# ani.save('dmpc_simulation.mp4', writer='ffmpeg', fps=10, dpi=200)
# print("저장 완료: dmpc_simulation.mp4")

# GIF로 저장 (화질 저하 및 파일 크기 증가 가능성 있음)
print("\nGIF 파일로 저장하는 중...")
ani.save('dmpc_simulation.gif', writer='pillow', fps=10)
print("저장 완료: dmpc_simulation.gif")

