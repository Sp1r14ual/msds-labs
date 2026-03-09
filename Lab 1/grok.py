import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


class Config:
    # 1. ОБЩИЕ ПАРАМЕТРЫ СИСТЕМЫ
    N = 20          # Количество шагов
    q = 1           # Количество точек в плане
    ki = [1]        # Количество наблюдений в каждой точке
    v = sum(ki)     # Общее число наблюдений на одном шаге

    # Размеры векторов
    n = 2           # размер состояния
    m = 1           # размер измерения
    r = 1           # размер управления
    p = 1           # размер шума процесса
    s = 2           # количество оцениваемых параметров θ

    # 2. ИСТИННЫЕ ЗНАЧЕНИЯ И ГРАНИЦЫ ПОИСКА
    theta_true = np.array([7.6, 39.5])
    bounds = Bounds([1.0, 30.0], [10.0, 45.0])

    # 3. ПОСТОЯННЫЕ МАТРИЦЫ И НАЧАЛЬНЫЕ УСЛОВИЯ
    G = np.array([[1.], [1.]])          # (2×1)
    H = np.array([[1., 0.]])             # (1×2)
    Q = np.array([[0.2]])                # (1×1)
    R = np.array([[0.5]])                # (1×1)
    P0 = np.array([[0.5, 0.], [0., 0.5]])
    x0 = np.zeros((n, 1))

    # 4. ФУНКЦИИ МАТРИЦ, ЗАВИСЯЩИХ ОТ THETA
    @staticmethod
    def get_F(theta):
        return np.array([[-theta[0], 1.],
                         [-39.5,     0.]])

    @staticmethod
    def get_Psi(theta):
        return np.array([[0.],
                         [theta[1]]])

    # 5. ГРАДИЕНТЫ МАТРИЦ ПО ПАРАМЕТРАМ (∂/∂θ₁, ∂/∂θ₂)
    dF = [
        np.array([[-1., 0.], [0., 0.]]),     # ∂F/∂θ₁
        np.array([[0., 0.],  [0., 0.]])      # ∂F/∂θ₂
    ]
    dPsi = [
        np.array([[0.], [0.]]),              # ∂Ψ/∂θ₁
        np.array([[0.], [1.]])               # ∂Ψ/∂θ₂
    ]

# Входной сигнал — константа 5 на всех шагах
Config.U = [[np.array([5.0]) for _ in range(Config.N)] for _ in range(Config.q)]

# Градиенты остальных (константных) матриц — нули
Config.dG  = [np.zeros((Config.n, Config.p)) for _ in range(Config.s)]
Config.dH  = [np.zeros((Config.m, Config.n)) for _ in range(Config.s)]
Config.dQ  = [np.zeros((Config.p, Config.p)) for _ in range(Config.s)]
Config.dR  = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
Config.dx0 = [np.zeros((Config.n, 1)) for _ in range(Config.s)]
Config.dP0 = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]


def ABAt(A, B):
    """ A @ B @ A.T """
    return A @ B @ A.T


def initY(theta, add_noise=True):
    """ Генерация измерений Y (с шумом или без) """
    F   = Config.get_F(theta)
    Psi = Config.get_Psi(theta)

    X = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]
    Y = [[[np.zeros((Config.m, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]

    flag = 1 if add_noise else 0

    for k in range(Config.N + 1):
        for i in range(Config.q):
            for j in range(Config.ki[i]):
                if k == 0:
                    X[k][i][j] = Config.x0.copy()
                else:
                    ui = Config.U[i][k-1].reshape(-1,1)   # (r,1) → здесь (1,1)
                    w = np.random.normal(0, np.sqrt(Config.Q)) if flag else 0
                    X[k][i][j] = F @ X[k-1][i][j] + Psi @ ui + Config.G * w

                v = np.random.normal(0, np.sqrt(Config.R)) if flag else 0
                Y[k][i][j] = Config.H @ X[k][i][j] + v

    return Y


def critIdent(theta, params):
    """ Отрицательный логарифм правдоподобия (критерий) """
    y = params['y']
    F   = Config.get_F(theta)
    Psi = Config.get_Psi(theta)

    Pkk = Config.P0.copy()
    x   = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]

    hi = Config.N * Config.m * Config.v * math.log(2 * math.pi)
    delta_sum = 0.0

    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.G, Config.Q)
        Bk1  = ABAt(Config.H, Pk1k) + Config.R
        invB = np.linalg.inv(Bk1)
        Kk1  = Pk1k @ Config.H.T @ invB
        Pk1k1 = (np.eye(Config.n) - Kk1 @ Config.H) @ Pk1k
        delta_k = 0.0
        for i in range(Config.q):
            ui = Config.U[i][k].reshape(-1,1)
            for j in range(Config.ki[i]):
                if k == 0:
                    xk = np.zeros((Config.n, 1))
                else:
                    xk = F @ x[k][i][j] + Psi @ ui

                eps = y[k+1][i][j] - Config.H @ xk
                x[k+1][i][j] = xk + Kk1 @ eps
                delta_k += (eps.T @ invB @ eps)[0,0]

        hi += Config.v * math.log(np.linalg.det(Bk1)) + delta_k
        Pkk = Pk1k1.copy()

    return hi / 2.0


def grad(theta, params):
    """ Аналитический градиент критерия """
    y = params['y']
    F   = Config.get_F(theta)
    Psi = Config.get_Psi(theta)

    Pkk  = Config.P0.copy()
    dPkk = [mat.copy() for mat in Config.dP0]

    x  = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]
    dx = [[[[np.zeros((Config.n, 1)) for _ in range(Config.s)] for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]

    dhi = np.zeros(Config.s)

    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.G, Config.Q)
        Bk1  = ABAt(Config.H, Pk1k) + Config.R
        invB = np.linalg.inv(Bk1)
        Kk1  = Pk1k @ Config.H.T @ invB
        Pk1k1 = (np.eye(Config.n) - Kk1 @ Config.H) @ Pk1k

        dBk  = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
        dK   = [np.zeros((Config.n, Config.m)) for _ in range(Config.s)]
        dPkk1= [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]

        for th in range(Config.s):
            dPk1k = Config.dF[th] @ Pkk @ F.T + F @ dPkk[th] @ F.T + F @ Pkk @ Config.dF[th].T
            dPk1k += Config.G @ Config.dQ[th] @ Config.G.T

            dB = Config.dH[th] @ Pk1k @ Config.H.T + Config.H @ dPk1k @ Config.H.T + Config.H @ Pk1k @ Config.dH[th].T + Config.dR[th]

            dK_num = dPk1k @ Config.H.T + Pk1k @ Config.dH[th].T
            dK[th] = dK_num @ invB - (Pk1k @ Config.H.T @ invB) @ dB @ invB

            dPkk1[th] = (np.eye(Config.n) - Kk1 @ Config.H) @ dPk1k
            dPkk1[th] -= (dK[th] @ Config.H + Kk1 @ Config.dH[th]) @ Pk1k

            dBk[th] = dB

        delta = np.zeros(Config.s)
        for i in range(Config.q):
            ui = Config.U[i][k].reshape(-1,1)
            for j in range(Config.ki[i]):
                if k == 0:
                    xk = np.zeros((Config.n, 1))
                    for th in range(Config.s):
                        dx[k][i][j][th] = Config.dx0[th].copy()
                else:
                    xk = F @ x[k][i][j] + Psi @ ui

                eps = y[k+1][i][j] - Config.H @ xk
                x[k+1][i][j] = xk + Kk1 @ eps

                for th in range(Config.s):
                    dxk = Config.dF[th] @ x[k][i][j] + F @ dx[k][i][j][th] + Config.dPsi[th] @ ui
                    deps = -Config.dH[th] @ xk - Config.H @ dxk
                    dx[k+1][i][j][th] = dxk + dK[th] @ eps + Kk1 @ deps

                    term1 = (deps.T @ invB @ eps)[0,0]
                    term2 = (eps.T @ invB @ dBk[th] @ invB @ eps)[0,0]
                    delta[th] += term1 - 0.5 * term2

        for th in range(Config.s):
            trace_term = np.trace(invB @ dBk[th])
            dhi[th] += (Config.v / 2.0) * trace_term + delta[th]

        Pkk  = Pk1k1.copy()
        dPkk = [dPkk1[th].copy() for th in range(Config.s)]

    return dhi


def main():
    num_experiments = 5
    thetaMean = np.zeros(Config.s)
    yMean = np.zeros(Config.N)
    yIdealMean = np.zeros(Config.N)

    for exp in range(num_experiments):
        Y = initY(Config.theta_true, add_noise=True)

        # случайное начальное приближение
        theta_start = np.random.uniform(Config.bounds.lb, Config.bounds.ub)

        res = minimize(
            fun=critIdent,
            jac=grad,
            x0=theta_start,
            args={'y': Y},
            bounds=Config.bounds,
            tol=1e-10,
            method='L-BFGS-B'          # обычно лучше всего работает здесь
        )
        print(f"Эксп. {exp+1:2d} → θ = {res.x.round(4)}  success={res.success}  nfev={res.nfev}")

        thetaMean += res.x

        # усреднённые измерения с шумом
        for k in range(1, Config.N + 1):
            yMean[k-1] += Y[k][0][0].item()

        # идеальная траектория по найденным параметрам
        Y_ideal = initY(res.x, add_noise=False)
        for k in range(1, Config.N + 1):
            yIdealMean[k-1] += Y_ideal[k][0][0].item()

    thetaMean /= num_experiments
    yMean /= num_experiments
    yIdealMean /= num_experiments

    sigma_param = np.linalg.norm(Config.theta_true - thetaMean) / np.linalg.norm(Config.theta_true)
    sigma_output = np.linalg.norm(yMean - yIdealMean) / np.linalg.norm(yMean)

    print("\n" + "═"*60)
    print(f"Средняя оценка: θ₁ = {thetaMean[0]:.4f},  θ₂ = {thetaMean[1]:.4f}")
    print(f"Истинные:       θ₁ = {Config.theta_true[0]},     θ₂ = {Config.theta_true[1]}")
    print(f"Ошибка в пространстве параметров  → {sigma_param:.6f}")
    print(f"Ошибка в пространстве откликов    → {sigma_output:.6f}")
    print("═"*60 + "\n")

    # График
    k = np.arange(0, Config.N)
    plt.figure(figsize=(10, 5.5))
    plt.yscale('symlog', linthresh=1.0)
    plt.plot(k, yMean, 'r-', lw=2.2, label='усреднённые измерения (с шумом)')
    plt.plot(k, yIdealMean, 'b--', lw=1.8, label='идеальная траектория по θ̂')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('k')
    plt.ylabel('y(k)')
    plt.title('Сравнение среднего отклика и восстановленной траектории')
    plt.legend()
    plt.tight_layout()
    plt.show()

main()