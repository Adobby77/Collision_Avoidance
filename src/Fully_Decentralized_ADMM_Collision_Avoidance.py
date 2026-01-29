import math
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

# ------------------------- EDITABLE SCENARIO ------------------------- #
SEED = 0
np.random.seed(SEED)

# params (no change)
H            = 25           # prediction horizon length
T            = 0.1          # discretization time step
MAX_ADMM     = 20           # ADMM iters per MPC step
K_adapt      = MAX_ADMM     # iterations to adapt linearization (Mod.1)
rho          = 10.0          # ADMM penalty weight
phi_deg      = 12.0         # rotation angle for deadlock (Mod.2)
detect_R     = 100.0         # neighbor detection radius
tol_p, tol_d = 1e-3, 1e-3

# physical limits
v_max      = 6.0
a_max      = 5.0

# cost function weight
Q = np.eye(2) * 1.0
R = np.eye(2) * 0.1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# params (scenario-dependent)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# flagship scenario (4 agents, crossing) - on paper

# M            = 4            # number of agents
# MAX_MPC      = 100           # MPC outer steps
# safety_R     = 10.0         # safety radius
# IS_COOP = [True, True, False, True]

# X0 = np.array([[-40, 40, 0, 0],   # x0 (global)
#                [0,  0,  40, -40]])  # y0 (global)
# V0 = np.array([[0, 0, 0, 0],   # vx (global)
#                [0, 0, 0, 0]])  # vy (global)
# G  = np.array([[40, -40, 0, 0],         # xd (global)
#                [0, 0, -40, 40]])        # yd (global)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# second scenario (6 agents, formation)

M            = 6            # number of agents
MAX_MPC      = 50           # MPC outer steps
safety_R     = 0.2          # safety radius
IS_COOP = [True, True, True, True, True, True]

X0 = np.array([[-3,  -5, -4,  -2, -6, -2],   # x0 (global)
               [-3,  -2,  -7, -5, -4, -6]])  # y0 (global)
V0 = np.array([[-0.5, 0.5, 0, -0.5, -0.5, 0],   # vx (global)
               [ 0.5, 0.5, 0.5, 0.5, -0.5, 0]])  # vy (global)
G  = np.array([[0, 1, 2, 1, 0, -1],         # xd (global)
               [0, 0, 1, 2, 2,  1]])        # yd (global)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 
# third scenario (4 agents, crossing)

# scenario: 6-agent formation cruising +x, 1 uncooperative intruder passing downward (-y)

# M            = 7             # 6 cooperative + 1 uncooperative
# MAX_MPC      = 80            # MPC outer steps
# safety_R     = 0.6           # tune to your scale
# IS_COOP      = [True, True, True, True, True, True, False]  # last agent is uncooperative

# # 6-agent formation in a 2x3 block, moving in +x; intruder starts above and passes straight down
# X0 = np.array([
#     [-8, -6, -4,  -8, -6, -4,   0],   # x0 (global): 6 in formation, intruder at x=0
#     [ 3,  2,  1,   1,  0, -1,   8]    # y0 (global): intruder starts at y=8
# ])

# # initial velocities: formation rolling forward; intruder descending quickly
# V0 = np.array([
#     [ 0.8, 0.8, 0.8,  0.8, 0.8, 0.8,  0.0],   # vx (global)
#     [ 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, -2.0]    # vy (global): intruder goes -y
# ])

# # goals: formation keeps lanes (same y), advances in +x; intruder exits below
# G = np.array([
#     [ 12, 14, 16,  12, 14, 16,   0],   # xd (global): shift formation forward ~20 units
#     [  3,  2,  1,   1,  0, -1,  -8]    # yd (global): intruder passes through to y=-8])

# ---------------------------- solver choice ---------------------------- #
def choose_solver():
    try:
        import gurobipy  # noqa: F401
        return cp.GUROBI
    except Exception:
        return cp.OSQP
SOLVER = choose_solver()

# ---------------------------- helpers (frame) -------------------------- #
def to_local(p_global, origin_global):
    """p_global(…,2) -> local wrt origin (pure translation)"""
    return p_global - origin_global

def from_local(p_local, origin_global):
    """local -> global wrt origin"""
    return p_local + origin_global

# ---------------------------- helpers (math) --------------------------- #
def time_shift(arr):
    """roll left by one along time axis"""
    return np.concatenate([arr[1:], arr[-1:]], axis=0)

def norm2_rows(A):
    return np.sqrt(np.sum(A*A, axis=1, keepdims=True))

def unit_rows(A, eps=1e-9):
    n = norm2_rows(A)
    return A / np.maximum(n, eps)

def predict_const_vel_local(goal_loc, H, T):
    """등속( a=0 ) 로컬에서 0 → goal_loc 직선 예측"""
    x = np.zeros((H+1, 2))
    v = np.zeros((H+1, 2))
    # 선형 보간 경로
    line = np.linspace(0.0, 1.0, H+1).reshape(-1,1)
    x = line * goal_loc[None,:]
    # 유한차분 속도
    v[:-1] = (x[1:] - x[:-1]) / T
    v[-1]  = v[-2]
    return x, v

def linearize_pair_i_local(xi, xj, safety_R):
    """
    i-로컬에서 지지 초평면:
      g_k = η_k^T((w_i[k]-w_ij[k]) - (xi[k]-xj[k])) + (||xi[k]-xj[k]|| - safety_R)
    """
    d = xi - xj                # (H+1,2)
    eta = unit_rows(d)
    base_gap = np.linalg.norm(d, axis=1) - safety_R  # (H+1,)
    def hbar(wi, wij):
        vals = []
        for k in range(xi.shape[0]):
            vals.append(eta[k].dot((wi[k]-wij[k]) - (xi[k]-xj[k])) + base_gap[k])
        return np.min(vals)
    return hbar, eta

def rotate_eta_list(eta, phi_rad):
    R = np.array([[np.cos(phi_rad), -np.sin(phi_rad)],
                  [np.sin(phi_rad),  np.cos(phi_rad)]])
    return (eta @ R.T)

def is_parallel_over_horizon(dx, eta, deg=5.0, eps=1e-9):
    # dx: (H,2) finite diff of x; eta: (H+1,2) align using eta[:-1]
    cos_th = []
    for k in range(dx.shape[0]):
        a = dx[k]; b = eta[k]
        na = max(np.linalg.norm(a), eps); nb = max(np.linalg.norm(b), eps)
        cosv = np.dot(a, b) / (na*nb)
        cos_th.append(cosv)
    return np.min(cos_th) > math.cos(math.radians(deg))

# --------- Collision checking helpers (logging) --------- #
def linear_slack_per_pair(ag_i, j):
    """에이전트 i(i-로컬)의 coordination 제약 g_k 벡터 반환."""
    if j not in ag_i.x_j or j not in ag_i.eta:
        return None
    xi, xj = ag_i.x, ag_i.x_j[j]
    eta = ag_i.eta[j]
    base_gap = np.linalg.norm(xi - xj, axis=1) - safety_R
    deltas = (ag_i.w - ag_i.w_arrow[j]) - (xi - xj)
    g = np.einsum('ij,ij->i', eta, deltas) + base_gap
    return g  # shape (H+1,)

def min_linear_slack_all(agents):
    gmin = np.inf; where = None
    for i, ag in enumerate(agents):
        for j in ag.x_j.keys():
            g = linear_slack_per_pair(ag, j)
            if g is None: continue
            m = float(np.min(g)); k = int(np.argmin(g))
            if m < gmin:
                gmin, where = m, (i, j, k)
    if gmin is np.inf: gmin = float('nan')
    return gmin, where

def min_true_clearance_pred(agents):
    """예측 궤적 w(전부 i-로컬이지만 차이는 불변)로 실제 거리여유 계산."""
    cmin = np.inf; where = None
    n = len(agents)
    # 전역 차이를 얻기 위해서는 아무 프레임이든 동일(순수 평행이동 불변)
    for i in range(n):
        wi = agents[i].w
        for j in range(i+1, n):
            # wi와 wj는 서로 다른 로컬 프레임이지만,
            # 전역 차이는 (wi_global - wj_global)로 계산해야 정확.
            # 간단히: w_i_global = wi + x0_i, w_j_global = wj + x0_j
            wj = agents[j].w
            dif = (wi + agents[i].x0) - (wj + agents[j].x0)  # (H+1,2)
            d = np.linalg.norm(dif, axis=1) - safety_R
            m = float(np.min(d)); k = int(np.argmin(d))
            if m < cmin:
                cmin, where = m, (i, j, k)
    if cmin is np.inf: cmin = float('nan')
    return cmin, where

def min_true_clearance_exec(traj_logs, safety_R):
    """실제로 실행된 MPC 포인트(전역) 기준 거리 여유."""
    M = len(traj_logs)
    if M == 0: return float('nan'), None
    Tm = min(len(t) for t in traj_logs)
    if Tm == 0: return float('nan'), None
    cmin = np.inf; where = None
    for i in range(M):
        xi = np.array(traj_logs[i])[:Tm]
        for j in range(i+1, M):
            xj = np.array(traj_logs[j])[:Tm]
            d = np.linalg.norm(xi - xj, axis=1) - safety_R
            m = float(np.min(d)); t = int(np.argmin(d))
            if m < cmin:
                cmin, where = m, (i, j, t)
    if cmin is np.inf: cmin = float('nan')
    return cmin, where

# ----------------- QP builders (i-로컬에서 정의) ---------------------- #
def build_double_integrator_QP_local(x_init_local, v_init, target_local,
                                     w_self, lamb_self,
                                     wj_recv, lamb_recv,
                                     Q, R, v_max, a_max, rho, H, T,
                                     lincons=None):
    """
    모든 변수는 에이전트 i의 로컬 프레임.
    lincons: list of (eta, xi_bar, xj_bar, base_gap) in i-local.
             제약: eta[k]^T((x[k]-xj_bar[k]) - (xi_bar[k]-xj_bar[k])) + base_gap[k] >= 0
    """
    x = cp.Variable((H+1, 2))
    v = cp.Variable((H+1, 2))
    a = cp.Variable((H,   2))

    cost = 0
    for k in range(H+1):
        cost += cp.quad_form(x[k]-target_local[k], Q)
    for k in range(H):
        cost += cp.quad_form(a[k], R)

    # Augmented Lagrangian terms
    cost += cp.sum(cp.multiply(lamb_self, (x - w_self))) + (rho/2.0)*cp.sum_squares(x - w_self)
    for j in wj_recv.keys():
        cost += cp.sum(cp.multiply(lamb_recv[j], (x - wj_recv[j]))) + (rho/2.0)*cp.sum_squares(x - wj_recv[j])

    # Dynamics & physical constraints
    cons = [x[0] == x_init_local, v[0] == v_init]
    for k in range(H):
        cons += [x[k+1] == x[k] + T*v[k],
                 v[k+1] == v[k] + T*a[k],
                 cp.norm_inf(v[k]) <= v_max,
                 cp.norm_inf(a[k]) <= a_max]

    # Linearized collision-avoidance (in i-local)
    if lincons:
        for (eta, xi_bar, xj_bar, base_gap) in lincons:
            for k in range(H+1):
                cons += [ eta[k] @ ((x[k]-xj_bar[k]) - (xi_bar[k]-xj_bar[k])) + float(base_gap[k]) >= 0 ]

    prob = cp.Problem(cp.Minimize(cost), cons)

    return prob, x, v, a

# ------------------------ Agent class (Fully Local) -------------------- #
class Agent:
    def __init__(self, idx, x0_global, v0_global, goal_global, is_coop=True):
        self.id = idx
        self.is_coop = is_coop
        self.x0 = x0_global.copy()   # GLOBAL origin of i-local
        self.v0 = v0_global.copy()   # velocity is translation invariant
        self.goal_global = goal_global.copy()

        # ADMM vars kept in i-LOCAL frame
        self.x = np.zeros((H+1,2))        # predicted pos (i-local), start at 0
        self.v = np.tile(self.v0, (H+1,1))
        self.a = np.zeros((H,2))
        self.w = self.x.copy()
        self.lmb = np.zeros_like(self.x)

        # neighbor / comm (all stored in i-local when attached to self)
        self.Nc, self.Nu = [], []
        self.rel = {}                      # GLOBAL offset: x0_j - x0_i  (2,)
        self.x_j = {}                      # j's predicted pos in i-local
        self.w_arrow = defaultdict(lambda: np.zeros_like(self.x))  # i->j proposal in i-local
        self.w_plus  = defaultdict(lambda: np.zeros_like(self.x))  # j->i proposal in i-local
        self.lmb_arrow = defaultdict(lambda: np.zeros_like(self.x))
        self.lmb_plus  = defaultdict(lambda: np.zeros_like(self.x))

        # linearization caches (i-local)
        self.hbar = {}
        self.eta  = {}
        self.hbar_changed = defaultdict(lambda: True)

        # for dual residuals
        self.w_prev = self.w.copy()
        self.w_arrow_prev = defaultdict(lambda: np.zeros_like(self.x))

    # ---------- neighbor detection (GLOBAL distance) ----------
    def detect_neighbors(self, agents):
        self.Nc.clear(); self.Nu.clear(); self.rel.clear()
        for other in agents:
            if other.id == self.id: continue
            d = other.x0 - self.x0  # GLOBAL
            if np.linalg.norm(d) <= detect_R:
                self.rel[other.id] = d.copy()  # GLOBAL
                if self.is_coop:
                    (self.Nc if other.is_coop else self.Nu).append(other.id)
                else:
                    # uncooperative i: treat all as Nu (no comm partners)
                    self.Nu.append(other.id)

    # ---------- communications (senders send LOCAL; receiver convert to i-local) ----------
    def receive_prediction(self, sender_id, x_sender_local):
        # x_sender_local : j-LOCAL, convert → i-LOCAL: + (x0_j - x0_i)
        d = self.rel[sender_id]  # GLOBAL (x0_j - x0_i)
        self.x_j[sender_id] = x_sender_local + d

    def receive_proposal(self, sender_id, w_sender_local, l_sender_local):
        # proposals from j: convert to i-local
        d = self.rel[sender_id]
        self.w_plus[sender_id]   = w_sender_local + d
        self.lmb_plus[sender_id] = l_sender_local  # translation-invariant for (x - w)

    # ---------- augmentation for uncooperative neighbors ----------
    def augment_uncoop_part1(self, agents):
        for j in self.Nu:
            aj = agents[j]
            # j의 등속 예측을 j-LOCAL에서 만들고 → i-LOCAL로 변환
            goal_j_local = aj.goal_global - aj.x0
            xj_loc, _ = predict_const_vel_local(goal_j_local, H, T)  # j-local
            d_ji = aj.x0 - self.x0                                   # GLOBAL
            self.x_j[j] = xj_loc + d_ji                              # i-local

    def augment_uncoop_part2(self):
        for j in self.Nu:
            # 비협력은 내 제안을 안 따르므로, 수신측 기준 w_plus[j] = x 로 유지(관측치 기반)
            self.w_plus[j] = self.x.copy()

    # ---------- targets in i-local ----------
    def target_traj_local(self):
        return np.tile(self.goal_global - self.x0, (H+1,1))

    # ---------- ADMM steps ----------
    def update_prediction(self):
        # 비협력: 로컬에서 0→goal 직선 (속도 제한), ADMM 항 없음
        if not self.is_coop:
            goal_loc = self.goal_global - self.x0
            xloc, vloc = predict_const_vel_local(goal_loc, H, T)
            self.x = xloc
            self.v = vloc
            self.a = np.zeros((H,2))
            self.w = self.x.copy(); self.lmb[:] = 0.0
            return

        target_loc = self.target_traj_local()

        # i-로컬에서 선형화 정보 구성 (Nc U Nu)
        neigh_ids = set(self.Nc) | set(self.Nu)
        lincons = []
        for j in neigh_ids:
            if j not in self.x_j:  # 아직 수신/추정 안되었으면 skip
                continue
            xj_bar = self.x_j[j]         # i-local
            xi_bar = self.x              # i-local (warm)
            if j not in self.eta:
                h_tmp, eta_tmp = linearize_pair_i_local(xi_bar, xj_bar, safety_R)
                self.hbar[j] = h_tmp
                self.eta[j]  = eta_tmp
            eta = self.eta[j]
            base_gap = np.linalg.norm(xi_bar - xj_bar, axis=1) - safety_R
            lincons.append((eta, xi_bar.copy(), xj_bar.copy(), base_gap.copy()))

        # QP (i-local)
        x0_local = np.zeros(2)  # 로컬 원점
        prob, x, v, a = build_double_integrator_QP_local(
            x0_local, self.v0, target_loc,
            self.w, self.lmb,
            self.w_plus, self.lmb_plus,
            Q, R, v_max, a_max, rho, H, T,
            lincons=None
        )
        prob.solve(solver=SOLVER, verbose=False)
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.x = x.value; self.v = v.value; self.a = a.value

    def update_linearization_and_deadlock(self, k_iter):
        if not self.is_coop:
            return
        if k_iter <= K_adapt:
            for j, xj in self.x_j.items():
                old = self.hbar.get(j, None)
                h, e = linearize_pair_i_local(self.x, xj, safety_R)
                self.hbar[j] = h
                self.eta[j]  = e
                self.hbar_changed[j] = (old is None)

        # deadlock test in i-local
        dx = self.x[1:] - self.x[:-1]
        for j in list(self.x_j.keys()):
            if (not self.hbar_changed[j]) and is_parallel_over_horizon(dx, self.eta[j]):
                self.eta[j] = rotate_eta_list(self.eta[j], math.radians(phi_deg))
                xi, xj = self.x, self.x_j[j]
                base_gap = np.linalg.norm(xi - xj, axis=1) - safety_R
                def hbar_rot(wi, wij, eta=self.eta[j], xi=xi, xj=xj, base_gap=base_gap):
                    vals = []
                    for k in range(xi.shape[0]):
                        vals.append(eta[k].dot((wi[k]-wij[k]) - (xi[k]-xj[k])) + base_gap[k])
                    return np.min(vals)
                self.hbar[j] = hbar_rot
                self.hbar_changed[j] = True

    def update_coordination(self):
        if not self.is_coop:
            return
        # i-로컬 QP: w, w_{i->j}
        w  = cp.Variable((H+1,2))
        wj = {j: cp.Variable((H+1,2)) for j in self.x_j.keys()}

        cost  = cp.sum(cp.multiply(self.lmb, (self.x - w))) + (rho/2.0)*cp.sum_squares(self.x - w)
        for j in self.x_j.keys():
            cost += cp.sum(cp.multiply(self.lmb_arrow[j], (self.x_j[j] - wj[j]))) \
                    + (rho/2.0)*cp.sum_squares(self.x_j[j] - wj[j])

        cons = []
        for j in self.x_j.keys():
            xi, xj = self.x, self.x_j[j]
            eta = self.eta[j]
            base_gap = np.linalg.norm(xi - xj, axis=1) - safety_R
            for k in range(H+1):
                cons.append( eta[k] @ ((w[k]-wj[j][k]) - (xi[k]-xj[k])) + base_gap[k] >= 0 )


        # uncooperative: w_{i->j} = x_j (고정)
        for j in self.Nu:
            if j in wj:
                cons.append( wj[j] == self.x_j[j] )

        prob = cp.Problem(cp.Minimize(cost), cons)

        prob.solve(solver=SOLVER, verbose=False)
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.w = w.value
            for j in self.x_j.keys():
                self.w_arrow[j] = wj[j].value

    def update_mediation(self):
        if not self.is_coop:
            return
        self.lmb = self.lmb + rho*(self.x - self.w)
        for j in self.x_j.keys():
            self.lmb_arrow[j] = self.lmb_arrow[j] + rho*(self.x_j[j] - self.w_arrow[j])

    # ---------- MPC apply & warm-start (rebase to new origin) ----------
    def rebase_local_all(self, delta_local):
        """원점이 x0 -> x0+Δ로 바뀔 때, 모든 '위치형' 로컬 필드에서 Δ를 빼줌."""
        # self-local fields
        self.x -= delta_local
        self.w -= delta_local
        # neighbor-attached fields in i-local
        for j in list(self.x_j.keys()):
            self.x_j[j]      -= delta_local
            self.w_arrow[j]  -= delta_local
            self.w_plus[j]   -= delta_local
            # lambdas are for (x-w), translation cancels out → 그대로 둠

    def apply_mpc_and_shift(self):
        if not self.is_coop:
            # straight step (global)
            d_global = self.goal_global - self.x0
            dist = np.linalg.norm(d_global)
            if dist > 1e-9:
                step = min(v_max*T, dist)
                diru = d_global / dist
                self.v0 = diru * (step/T)               # global
                self.x0 = self.x0 + diru*step           # global
            # warm-start (local): shift & rebase by predicted delta (old x[1])
            delta = self.x[1].copy() if self.x.shape[0] > 1 else np.zeros(2)
            self.x = time_shift(self.x)
            self.v = time_shift(self.v)
            if self.a.shape[0] > 0:
                self.a = np.vstack([self.a[1:], self.a[-1:]])
            self.w = time_shift(self.w)
            self.rebase_local_all(delta)
            self.w_prev = self.w.copy()
            for j in self.x_j.keys():
                self.w_arrow_prev[j] = self.w_arrow[j].copy()
            return

        # cooperative: apply a0 (global equal to local)
        a0 = self.a[0] if self.a.shape[0] > 0 else np.zeros(2)
        self.x0 = self.x0 + T*self.v0       # global
        self.v0 = self.v0 + T*a0            # global (translation invariant)

        # warm-start shift in local + rebase by predicted delta (old x[1])
        delta = self.x[1].copy() if self.x.shape[0] > 1 else np.zeros(2)

        self.x = time_shift(self.x)
        self.v = time_shift(self.v)
        if self.a.shape[0] > 0:
            self.a = np.vstack([self.a[1:], self.a[-1:]])
        self.w = time_shift(self.w)

        # shift all neighbor-attached locals as well
        for j in list(self.x_j.keys()):
            self.x_j[j]     = time_shift(self.x_j[j])
            self.w_arrow[j] = time_shift(self.w_arrow[j])
            self.w_plus[j]  = time_shift(self.w_plus[j])
            self.lmb_arrow[j] = time_shift(self.lmb_arrow[j])
            self.lmb_plus[j]  = time_shift(self.lmb_plus[j])

        # rebase by delta so new origin is 0
        self.rebase_local_all(delta)

        # store prevs for dual residuals
        self.w_prev = self.w.copy()
        for j in self.x_j.keys():
            self.w_arrow_prev[j] = self.w_arrow[j].copy()

# ------------------------------- Sim loop ------------------------------- #
def compute_min_linear_slack(agents):
    min_slack = float('inf')
    for ag in agents:
        for j in ag.x_j.keys():
            try:
                val = ag.hbar[j](ag.w, ag.w_arrow[j])  # i-local
                if val < min_slack: min_slack = val
            except Exception:
                pass
    if min_slack == float('inf'):
        min_slack = float('nan')
    return min_slack

def residuals_below(agents, tol_p, tol_d):
    rp = []; rd = []
    for ag in agents:
        rp.append(np.max(np.abs(ag.x - ag.w)))
        for j in ag.x_j.keys():
            rp.append(np.max(np.abs(ag.x - ag.w_plus[j])))
        rd.append(rho * np.max(np.abs(ag.w - ag.w_prev)))
        for j in ag.x_j.keys():
            rd.append(rho * np.max(np.abs(ag.w_arrow[j] - ag.w_arrow_prev[j])))
    return (max(rp) <= tol_p) and (max(rd) <= tol_d)

def main():
    agents = [Agent(i, X0[:,i], V0[:,i], G[:,i], IS_COOP[i]) for i in range(M)]
    traj_logs = [[] for _ in range(M)]   # GLOBAL executed positions per MPC step

    for t in range(MAX_MPC):  # MPC outer loop
        print(f"\n[MPC step {t}] starting...")
        
        # neighbor detection (GLOBAL)
        for ag in agents:
            ag.detect_neighbors(agents)

        # reset prev buffers (for dual residuals)
        for ag in agents:
            ag.w_prev = ag.w.copy()
            ag.w_arrow_prev = defaultdict(lambda: np.zeros_like(ag.x))

        # ADMM iterations (Algorithm 3 order)
        for k in range(1, MAX_ADMM+1):
            if k % 10 == 1:
                print(f"  ADMM iter {k}...")

            # 1) Prediction (i-local, uses last w, λ, w_plus)
            for ag in agents:
                ag.update_prediction()

            # * Comm 1: send x (LOCAL) to cooperative neighbors
            for s in agents:
                if not s.is_coop: continue
                xs_local = s.x  # s-local
                for rid in s.Nc:
                    agents[rid].receive_prediction(s.id, xs_local)

            # + Augment 1: uncooperative neighbors (i-local)
            for ag in agents:
                ag.augment_uncoop_part1(agents)

            # + Adaptive linearization & Deadlock protection (i-local)
            for ag in agents:
                ag.update_linearization_and_deadlock(k)

            # 2) Coordination (i-local; includes linearized collision constraints)
            for ag in agents:
                ag.update_coordination()

            # 3) Mediation (i-local)
            for ag in agents:
                ag.update_mediation()

            # * Comm 2: send (w_{i->j}, λ_{i->j}) (i-local) → receiver converts to j-local internally
            for s in agents:
                if not s.is_coop: continue
                for rid in s.Nc:
                    agents[rid].receive_proposal(s.id, s.w_arrow[rid], s.lmb_arrow[rid])

            # + Augment 2: uncooperative assume w_plus := x (already in method)
            for ag in agents:
                ag.augment_uncoop_part2()

            # periodic logging
            if (k % 10 == 0) or (k == MAX_ADMM):
                rp = []; rd = []
                for ag in agents:
                    rp.append(np.max(np.abs(ag.x - ag.w)))
                    for j in ag.x_j.keys():
                        rp.append(np.max(np.abs(ag.x - ag.w_plus[j])))
                    rd.append(rho * np.max(np.abs(ag.w - ag.w_prev)))
                    for j in ag.x_j.keys():
                        rd.append(rho * np.max(np.abs(ag.w_arrow[j] - ag.w_arrow_prev[j])))
                min_slack = compute_min_linear_slack(agents)
                lin_min, lin_where = min_linear_slack_all(agents)
                true_min_pred, true_where_pred = min_true_clearance_pred(agents)
                print(f"    rp_max={max(rp):.3e}, rd_max={max(rd):.3e}, "
                      f"min_hbar={min_slack:.3e}")
                print(f"    lin_slack_min={lin_min:.3e} at {lin_where or '-'}; "
                      f"true_clear_min(pred)={true_min_pred:.3e} at {true_where_pred or '-'}")
                if (np.isfinite(lin_min) and lin_min < -1e-6) or \
                   (np.isfinite(true_min_pred) and true_min_pred < -1e-6):
                    print("    !!! WARNING: potential collision (negative slack/clearance) !!!")

            if residuals_below(agents, tol_p, tol_d):
                print(f"  → converged at iter {k}")
                break

        # apply MPC step & rebase locals; log executed global pos
        for m, ag in enumerate(agents):
            traj_logs[m].append(ag.x0.copy())  # executed global
            ag.apply_mpc_and_shift()

    plot_and_animate(traj_logs)

# ------------------------ Plot & Animation ------------------------ #

def _axes_style(ax, dark=True):
    if dark:
        plt.style.use('dark_background')
        bg, fg, grid = ('#0e1117', '#e6edf3', (1,1,1,0.08))
    else:
        plt.style.use('default')
        bg, fg, grid = ('#ffffff', '#111111', (0,0,0,0.08))

    ax.set_facecolor(bg)
    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6, color=grid)
    ax.set_aspect('equal')
    ax.set_xlabel('X', fontsize=12, color=fg)
    ax.set_ylabel('Y', fontsize=12, color=fg)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_alpha(0.2)
    return fg

def _compute_bounds(traj_logs, pad_ratio=0.12):
    trajs = [np.array(t) for t in traj_logs if len(t) > 0]
    A = np.vstack(trajs + [X0.T, G.T]) if trajs else np.vstack([X0.T, G.T])
    (xmin, ymin), (xmax, ymax) = A.min(axis=0), A.max(axis=0)
    size = max(xmax - xmin, ymax - ymin)
    pad = max(1e-6, size * pad_ratio)
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad

def plot_static_results(traj_logs, colors, dark=True, show_ids=True):
    """정적 결과 플롯 (깔끔한 마커, 라벨, 최소거리 주석 포함)"""
    fig, ax = plt.subplots(figsize=(11, 8))
    fg = _axes_style(ax, dark=dark)

    # 궤적
    for m in range(M):
        tr = np.array(traj_logs[m]) if len(traj_logs[m]) else np.array([X0[:,m]])
        c  = colors[m]
        tag = 'uncoop' if not IS_COOP[m] else 'coop'
        # 꼬리 두께 가변 라인
        if len(tr) > 1:
            from matplotlib.collections import LineCollection
            seg = np.concatenate([tr[:-1,None,:], tr[1:,None,:]], axis=1)
            lc = LineCollection(seg, linewidths=np.linspace(2, 4, len(seg)),
                                colors=[c]*len(seg), alpha=0.55)
            ax.add_collection(lc)
        ax.plot(tr[:,0], tr[:,1], '-', color=c, alpha=0.25)

        # 시작/목표 마커
        ax.scatter(X0[0,m], X0[1,m], marker='D', s=80, color=c, ec='black', lw=0.8, zorder=5)
        ax.scatter(G[0,m],  G[1,m],  marker='*', s=140, color=c, ec='black', lw=0.8, zorder=5)

        # 에이전트 라벨
        if show_ids:
            ax.text(tr[-1,0], tr[-1,1], f' {m}({tag})',
                    fontsize=10, color=fg, ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=0.25',
                              fc=(0,0,0,0.35) if dark else (1,1,1,0.55),
                              ec=(1,1,1,0.2), lw=0.5))

    # 안전반경 안내(범례용 더미)
    from matplotlib.lines import Line2D
    leg_items = [
        Line2D([0],[0], marker='D', color='none', markerfacecolor=fg, markeredgecolor='black',
               markersize=8, label='Start'),
        Line2D([0],[0], marker='*', color='none', markerfacecolor=fg, markeredgecolor='black',
               markersize=12, label='Goal'),
    ]
    ax.legend(handles=leg_items, loc='best', fontsize=10, framealpha=0.2)

    # 축 범위
    xmin, xmax, ymin, ymax = _compute_bounds(traj_logs)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title('ADMM–DMPC Trajectories (Pretty Static)', fontsize=16, color=fg)

    # 최소 실제 분리거리 주석
    cmin, where = min_true_clearance_exec(traj_logs, safety_R)
    if np.isfinite(cmin):
        ax.text(0.02, 0.98, f'min clearance = {cmin:.3f}',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=11, color=('#ffb4a2' if cmin < 0 else '#a2ffb4'),
                bbox=dict(boxstyle='round,pad=0.25',
                          fc=(0.25,0,0,0.35) if cmin<0 else (0,0.25,0,0.35),
                          ec='none'))

    plt.tight_layout()
    fig.savefig("admm_dmpc_static_pretty.png", dpi=200)
    print("Saved static plot: admm_dmpc_static_pretty.png")


def create_animation(traj_logs, colors, dark=True, tail_sec=2.0, save=True):
    """부드러운 꼬리/아이콘/속도벡터가 있는 예쁜 애니메이션"""
    trajs = [np.array(t) for t in traj_logs]
    if not trajs or min(len(t) for t in trajs) <= 1:
        print("No trajectory data to animate."); return

    Tm = min(len(t) for t in trajs)
    xmin, xmax, ymin, ymax = _compute_bounds(traj_logs)

    # Figure & Axes
    fig, ax = plt.subplots(figsize=(11, 8))
    _ = _axes_style(ax, dark=dark)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # 시작/목표 마커
    for m in range(M):
        c = colors[m]
        ax.scatter(X0[0,m], X0[1,m], marker='D', s=80, color=c, ec='white', lw=0.8, alpha=0.9, zorder=3)
        ax.scatter(G[0,m],  G[1,m],  marker='*', s=140, color=c, ec='white', lw=0.8, alpha=0.9, zorder=3)

    # 에이전트 바디만 표시 (안전반경 제거)
    agent_body = [Circle((0,0), 0.1, fc=colors[i], ec='white', lw=0.7, alpha=0.95)
                  for i in range(M)]
    for b in agent_body:
        ax.add_patch(b)

    # 꼬리(그라데이션) 준비
    tail_len = max(1, int(round(tail_sec / T)))
    trail_collections = []
    for i in range(M):
        lc = LineCollection([], linewidths=2.0, alpha=0.9)
        ax.add_collection(lc)
        trail_collections.append(lc)

    # 속도 벡터
    q = ax.quiver([0]*M, [0]*M, [0]*M, [0]*M, angles='xy', scale_units='xy',
                  scale=1.0, width=0.004, color=colors, alpha=0.9)

    # Coop/Uncoop 배지
    badges = []
    for i in range(M):
        badges.append(ax.text(0, 0, 'C' if IS_COOP[i] else 'U',
                              fontsize=9, color='black',
                              ha='center', va='center',
                              bbox=dict(boxstyle='circle,pad=0.28',
                                        fc='#a2ffb4' if IS_COOP[i] else '#ffb4a2',
                                        ec='white', lw=0.8, alpha=0.9), zorder=6))

    def init():
        for i in range(M):
            p0 = trajs[i][0]
            agent_body[i].center = p0
            trail_collections[i].set_segments([])
            badges[i].set_position((p0[0], p0[1]))
        q.set_offsets(np.array([t[0] for t in trajs]))
        q.set_UVC(np.zeros(M), np.zeros(M))
        ax.set_title('ADMM–DMPC Animation', fontsize=16)
        return [*agent_body, *trail_collections, q, *badges]

    def update(frame):
        # 위치/꼬리
        for i in range(M):
            pos = trajs[i][frame]
            agent_body[i].center = pos

            s_idx = max(0, frame - tail_len)
            tail  = trajs[i][s_idx:frame+1]
            if len(tail) > 1:
                pts = tail.reshape(-1,1,2)
                seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
                trail_collections[i].set_segments(seg)
                alphas = np.linspace(0.1, 0.95, len(seg))
                widths = np.linspace(1.2, 3.2, len(seg))
                trail_collections[i].set_color([(*colors[i][:3], a) for a in alphas])
                trail_collections[i].set_linewidth(widths)

        # 속도 벡터
        P = np.array([trajs[i][frame] for i in range(M)])
        if frame > 0:
            Pm1 = np.array([trajs[i][frame-1] for i in range(M)])
            V    = (P - Pm1) / T
        else:
            V = V0.T.copy()

        # 축 스팬의 일정 비율(예: 5%)을 목표 길이로
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        span   = max(x1 - x0, y1 - y0)
        target_len = 0.05 * span          # <- 원하는 비율로 조절 (0.03~0.06 권장)

        # 프레임 내 최대 속도를 target_len에 맞춰 정규화
        speed = np.linalg.norm(V, axis=1)
        vmax  = max(np.max(speed), 1e-9)
        scale = target_len / vmax
        Vdisp = V * scale

        q.set_offsets(P)
        q.set_UVC(Vdisp[:,0], Vdisp[:,1])


        # 배지 위치
        for i in range(M):
            px, py = P[i]
            badges[i].set_position((px, py))  # safety_R 대신 고정값 사용

        ax.set_title(f'ADMM–DMPC Animation  |  t = {frame*T:.2f}s', fontsize=16)
        return [*agent_body, *trail_collections, q, *badges]

    fps = max(6, int(round(1.0 / T)))
    ani = animation.FuncAnimation(fig, update, frames=Tm, init_func=init, blit=False, interval=1000/fps)

    if save:
        try:
            print('Saving GIF: admm_simulation.gif...')
            ani.save('admm_simulation.gif', writer='pillow', fps=fps)
            print('Saved GIF: admm_simulation.gif')
        except Exception as e:
            print(f'GIF save failed: {e}')
    # plt.show()



def plot_and_animate(traj_logs, dark=True):
    """정적 플롯 → 애니메이션 순서로 예쁘게 출력"""
    colors = plt.cm.viridis(np.linspace(0, 1, M))
    import matplotlib.colors as mcolors
    colors[0] = mcolors.to_rgba('#FF7F0E', 1.0)
    plot_static_results(traj_logs, colors, dark=dark, show_ids=True)
    create_animation(traj_logs, colors, dark=dark, tail_sec=2.0, save=True)
# ======================================================================


# GO!
if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\n[Done] Total wall-clock time = {time.perf_counter() - t0:.3f} s")