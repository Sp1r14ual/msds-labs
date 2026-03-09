import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


class Config:
    # 1. ОБЩИЕ ПАРАМЕТРЫ
    N = 20                  # число шагов
    q = 1                   # число точек плана
    ki = [1]                # число наблюдений в каждой точке
    v = sum(ki)             # общее число наблюдений на шаге

    n = 2                   # размер состояния x
    m = 1                   # размер измерения y
    r = 1                   # размер управления u
    p = 1                   # размер шума системы w
    s = 2                   # число неизвестных параметров theta

    # 2. ИСТИННЫЕ ПАРАМЕТРЫ И ГРАНИЦЫ
    theta_true = np.array([7.6, 39.5], dtype=float)
    bounds = Bounds([1.0, 30.0], [10.0, 45.0])

    # 3. ПОСТОЯННЫЕ МАТРИЦЫ
    Gamma = np.array([[1.0],
                      [1.0]])

    H = np.array([[1.0, 0.0]])

    Q = np.array([[0.2]])
    R = np.array([[0.5]])

    P0 = np.array([[0.5, 0.0],
                   [0.0, 0.5]])

    x0 = np.zeros((n, 1))

    # Константа из F
    f21_const = -39.5

    # 4. МАТРИЦЫ, ЗАВИСЯЩИЕ ОТ theta
    @staticmethod
    def get_F(theta):
        theta1 = theta[0]
        return np.array([[-theta1, 1.0],
                         [Config.f21_const, 0.0]])

    @staticmethod
    def get_Psi(theta):
        theta2 = theta[1]
        return np.array([[0.0],
                         [theta2]])

    # 5. ПРОИЗВОДНЫЕ ПО theta1, theta2
    dF = [
        np.array([[-1.0, 0.0],
                  [ 0.0, 0.0]]),
        np.zeros((2, 2))
    ]

    dPsi = [
        np.zeros((2, 1)),
        np.array([[0.0],
                  [1.0]])
    ]


# Постоянный вход U^T = [5, ..., 5]
Config.U = [[np.array([5.0], dtype=float) for _ in range(Config.N)]
            for _ in range(Config.q)]

# Производные константных матриц
Config.dGamma = [np.zeros((Config.n, Config.p)) for _ in range(Config.s)]
Config.dH     = [np.zeros((Config.m, Config.n)) for _ in range(Config.s)]
Config.dQ     = [np.zeros((Config.p, Config.p)) for _ in range(Config.s)]
Config.dR     = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
Config.dx0    = [np.zeros((Config.n, 1)) for _ in range(Config.s)]
Config.dP0    = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]


def ABAt(A, B):
    """Вычисляет A * B * A^T."""
    return A @ B @ A.T


def initY(theta, with_noise=True):
    """Генерация выходов системы."""
    F = Config.get_F(theta)
    Psi = Config.get_Psi(theta)
    noise_flag = 1 if with_noise else 0

    X = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])]
          for i in range(Config.q)] for _ in range(Config.N + 1)]

    Y = [[[np.zeros((Config.m, 1)) for _ in range(Config.ki[i])]
          for i in range(Config.q)] for _ in range(Config.N + 1)]

    for k in range(Config.N + 1):
        for i in range(Config.q):
            for j in range(Config.ki[i]):
                if k == 0:
                    X[k][i][j] = Config.x0.copy()
                    v = np.random.normal(
                        0.0,
                        np.sqrt(Config.R[0, 0]),
                        (Config.m, 1)
                    )
                    Y[k][i][j] = Config.H @ X[k][i][j] + noise_flag * v
                else:
                    uk = Config.U[i][k - 1].reshape(Config.r, 1)

                    w = np.random.normal(
                        0.0,
                        np.sqrt(Config.Q[0, 0]),
                        (Config.p, 1)
                    )

                    X[k][i][j] = (
                        F @ X[k - 1][i][j]
                        + Psi @ uk
                        + noise_flag * (Config.Gamma @ w)
                    )

                    v = np.random.normal(
                        0.0,
                        np.sqrt(Config.R[0, 0]),
                        (Config.m, 1)
                    )

                    Y[k][i][j] = Config.H @ X[k][i][j] + noise_flag * v

    return Y


def critIdent(theta, params):
    """Критерий идентификации."""
    y = params["y"]

    F = Config.get_F(theta)
    Psi = Config.get_Psi(theta)

    Pkk = Config.P0.copy()

    x = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])]
          for i in range(Config.q)] for _ in range(Config.N + 1)]

    hi = Config.N * Config.m * Config.v * math.log(2.0 * math.pi)

    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.Gamma, Config.Q)
        Bk1 = ABAt(Config.H, Pk1k) + Config.R
        Bk1_inv = np.linalg.inv(Bk1)

        Kk1 = Pk1k @ Config.H.T @ Bk1_inv
        Pk1k1 = (np.eye(Config.n) - Kk1 @ Config.H) @ Pk1k

        delta = 0.0

        for i in range(Config.q):
            uk = Config.U[i][k].reshape(Config.r, 1)

            for j in range(Config.ki[i]):
                if k == 0:
                    x[k][i][j] = Config.x0.copy()

                xk1k = F @ x[k][i][j] + Psi @ uk
                epsk1 = y[k + 1][i][j] - Config.H @ xk1k
                x[k + 1][i][j] = xk1k + Kk1 @ epsk1

                delta += (epsk1.T @ Bk1_inv @ epsk1)[0, 0]

        hi += Config.v * math.log(np.linalg.det(Bk1)) + delta
        Pkk = Pk1k1.copy()

    return hi / 2.0


def grad(theta, params):
    """Аналитический градиент критерия идентификации."""
    y = params["y"]

    F = Config.get_F(theta)
    Psi = Config.get_Psi(theta)

    Pkk = Config.P0.copy()
    dPkk = [mat.copy() for mat in Config.dP0]

    x = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])]
          for i in range(Config.q)] for _ in range(Config.N + 1)]

    dx = [[[[np.zeros((Config.n, 1)) for _ in range(Config.s)]
            for _ in range(Config.ki[i])]
           for i in range(Config.q)] for _ in range(Config.N + 1)]

    dhi = np.zeros(Config.s)

    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.Gamma, Config.Q)
        Bk1 = ABAt(Config.H, Pk1k) + Config.R
        Bk1_inv = np.linalg.inv(Bk1)

        Kk1 = Pk1k @ Config.H.T @ Bk1_inv
        Pk1k1 = (np.eye(Config.n) - Kk1 @ Config.H) @ Pk1k

        dBk1 = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
        dKk1 = [np.zeros((Config.n, Config.m)) for _ in range(Config.s)]
        dPk1k1 = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]

        for th in range(Config.s):
            dPk1k = Config.dF[th] @ Pkk @ F.T
            dPk1k += ABAt(F, dPkk[th]) + ABAt(Config.Gamma, Config.dQ[th])
            dPk1k += F @ Pkk @ Config.dF[th].T
            dPk1k += Config.dGamma[th] @ Config.Q @ Config.Gamma.T
            dPk1k += Config.Gamma @ Config.Q @ Config.dGamma[th].T

            dBk1[th] = Config.dH[th] @ Pk1k @ Config.H.T
            dBk1[th] += ABAt(Config.H, dPk1k)
            dBk1[th] += Config.H @ Pk1k @ Config.dH[th].T + Config.dR[th]

            dKk1[th] = dPk1k @ Config.H.T + Pk1k @ Config.dH[th].T
            dKk1[th] -= Pk1k @ Config.H.T @ Bk1_inv @ dBk1[th]
            dKk1[th] = dKk1[th] @ Bk1_inv

            dPk1k1[th] = (np.eye(Config.n) - Kk1 @ Config.H) @ dPk1k
            dPk1k1[th] -= (dKk1[th] @ Config.H + Kk1 @ Config.dH[th]) @ Pk1k

        delta = np.zeros(Config.s)

        for i in range(Config.q):
            uk = Config.U[i][k].reshape(Config.r, 1)

            for j in range(Config.ki[i]):
                if k == 0:
                    x[k][i][j] = Config.x0.copy()
                    for th in range(Config.s):
                        dx[k][i][j][th] = Config.dx0[th].copy()

                xk1k = F @ x[k][i][j] + Psi @ uk
                epsk1 = y[k + 1][i][j] - Config.H @ xk1k
                x[k + 1][i][j] = xk1k + Kk1 @ epsk1

                for th in range(Config.s):
                    dxk1k = (
                        Config.dF[th] @ x[k][i][j]
                        + F @ dx[k][i][j][th]
                        + Config.dPsi[th] @ uk
                    )

                    depsk1 = -Config.dH[th] @ xk1k - Config.H @ dxk1k

                    dx[k + 1][i][j][th] = (
                        dxk1k
                        + dKk1[th] @ epsk1
                        + Kk1 @ depsk1
                    )

                    delta[th] += (depsk1.T @ Bk1_inv @ epsk1)[0, 0]
                    delta[th] -= 0.5 * (
                        epsk1.T @ Bk1_inv @ dBk1[th] @ Bk1_inv @ epsk1
                    )[0, 0]

        for th in range(Config.s):
            dhi[th] += 0.5 * Config.v * np.trace(Bk1_inv @ dBk1[th]) + delta[th]

        Pkk = Pk1k1.copy()
        dPkk = dPk1k1.copy()

    return np.array(dhi, dtype=float)


def extract_mean_output(Y):
    """Средний выход по наблюдениям в каждой точке плана."""
    out = np.zeros((Config.N + 1, Config.q))

    for k in range(Config.N + 1):
        for i in range(Config.q):
            vals = [Y[k][i][j][0, 0] for j in range(Config.ki[i])]
            out[k, i] = np.mean(vals)

    return out


def main():
    # Для воспроизводимости можно раскомментировать:
    # np.random.seed(42)

    num_experiments = 5

    theta_sum = np.zeros(Config.s)
    y_sum = np.zeros((Config.N + 1, Config.q))

    for exp_id in range(num_experiments):
        Y = initY(Config.theta_true, with_noise=True)

        theta_start = np.array([
            np.random.uniform(Config.bounds.lb[0], Config.bounds.ub[0]),
            np.random.uniform(Config.bounds.lb[1], Config.bounds.ub[1])
        ])

        result = minimize(
            fun=critIdent,
            jac=grad,
            x0=theta_start,
            args=({"y": Y},),      # важно: args передаем кортежем
            method="L-BFGS-B",
            bounds=Config.bounds,
            tol=1e-10
        )

        print(f"Эксперимент {exp_id + 1}, оценка theta: {result.x}")

        theta_sum += result.x
        y_sum += extract_mean_output(Y)

    theta_mean = theta_sum / num_experiments
    y_mean = y_sum / num_experiments

    sigma1 = np.linalg.norm(Config.theta_true - theta_mean) / np.linalg.norm(Config.theta_true)

    Y_ideal = initY(theta_mean, with_noise=False)
    y_ideal = extract_mean_output(Y_ideal)

    sigma2 = np.linalg.norm(y_mean - y_ideal) / np.linalg.norm(y_mean)

    print("-" * 40)
    print(f"Средняя оценка theta: {theta_mean}")
    print(f"Ошибка в пространстве параметров sigma1 = {sigma1:.6f}")
    print(f"Ошибка в пространстве откликов   sigma2 = {sigma2:.6f}")

    # График
    x_axis = np.arange(Config.N + 1)

    plt.figure(figsize=(10, 6))
    plt.yscale("symlog", linthresh=1.0)

    plt.plot(x_axis, y_mean[:, 0], linewidth=3, label="Y_cp (с шумом, усредненное)")
    plt.plot(x_axis, y_ideal[:, 0], linestyle="--", linewidth=2, label="Y* (без шума)")

    plt.xlabel("k (шаг)")
    plt.ylabel("Y")
    plt.title("Графики Y и Y*")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()