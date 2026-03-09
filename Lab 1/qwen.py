import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds

class Config:
    # =====================================================================
    # 1. ОБЩИЕ ПАРАМЕТРЫ СИСТЕМЫ
    # =====================================================================
    N = 20                          # Количество шагов
    q = 1                           # Количество точек в плане
    ki = [1]                        # Количество наблюдений в каждой точке
    v = sum(ki)                     # Общее число наблюдений на одном шаге
    
    # Размеры векторов
    n = 2                           # Размер вектора состояния x
    m = 1                           # Размер вектора измерений y
    r = 1                           # Размер вектора управления u
    p = 1                           # Размер вектора шума системы w
    s = 2                           # Количество неизвестных параметров theta
    
    # =====================================================================
    # 2. ИСТИННЫЕ ЗНАЧЕНИЯ И ГРАНИЦЫ ПОИСКА
    # =====================================================================
    theta_true = np.array([7.6, 39.5])
    bounds = Bounds([1.0, 30.0], [10.0, 45.0])
    
    # =====================================================================
    # 3. ПОСТОЯННЫЕ МАТРИЦЫ И НАЧАЛЬНЫЕ УСЛОВИЯ
    # =====================================================================
    G = np.array([[1], [1]])
    H = np.array([[1, 0]])
    Q = np.array([[0.2]])
    R = np.array([[0.5]])
    P0 = np.array([[0.5, 0], [0, 0.5]])
    x0 = np.zeros((n, 1))
    
    # =====================================================================
    # 4. ГРАДИЕНТЫ МАТРИЦ ПО ПАРАМЕТРАМ THETA
    # =====================================================================
    dF = [np.array([[-1, 0], [0, 0]]), 
          np.array([[0, 0], [0, 0]])]
    
    dPsi = [np.array([[0], [0]]), 
            np.array([[0], [1]])]
    
    # Градиенты для константных матриц (будут инициализированы после класса)
    dG = None
    dH = None
    dQ = None
    dR = None
    dx0 = None
    dP0 = None
    
    # Управление (будет инициализировано после класса)
    U = None
    
    # =====================================================================
    # 5. ФУНКЦИИ МАТРИЦ, ЗАВИСЯЩИХ ОТ ПАРАМЕТРОВ THETA
    # =====================================================================
    @staticmethod
    def get_F(theta):
        return np.array([[-theta[0], 1], 
                         [-39.5, 0]])
    
    @staticmethod
    def get_Psi(theta):
        return np.array([[0], 
                         [theta[1]]])
    
    @staticmethod
    def init_dependent_vars():
        """Инициализация переменных, зависящих от параметров класса"""
        Config.U = [[5.0 * np.ones(Config.r) for _ in range(Config.N)] 
                    for _ in range(Config.q)]
        
        Config.dG = [np.zeros((Config.n, Config.p)) for _ in range(Config.s)]
        Config.dH = [np.zeros((Config.m, Config.n)) for _ in range(Config.s)]
        Config.dQ = [np.zeros((Config.p, Config.p)) for _ in range(Config.s)]
        Config.dR = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
        Config.dx0 = [np.zeros((Config.n, 1)) for _ in range(Config.s)]
        Config.dP0 = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]


# =========================================================================
# ИНИЦИАЛИЗАЦИЯ ЗАВИСИМЫХ ПЕРЕМЕННЫХ ПОСЛЕ СОЗДАНИЯ КЛАССА
# =========================================================================
Config.init_dependent_vars()


def ABAt(A, B):
    """Вычисление A * B * A^T"""
    return A @ B @ A.T


def initY(theta, flag):
    """
    Генерация выхода системы (измерений Y)
    flag=1: с шумом, flag=0: без шума
    """
    F = Config.get_F(theta)
    psi = Config.get_Psi(theta)
    a = 1 if flag == 1 else 0
    
    X = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] 
          for i in range(Config.q)] for _ in range(Config.N + 1)]
    Y = [[[np.zeros((Config.m, 1)) for _ in range(Config.ki[i])] 
          for i in range(Config.q)] for _ in range(Config.N + 1)]
    
    for k in range(Config.N + 1):
        for i in range(Config.q):
            for j in range(Config.ki[i]):
                if k == 0:
                    X[k][i][j] = Config.x0.copy()
                    v = np.random.normal(0, np.sqrt(Config.R), (Config.m, 1))
                    Y[k][i][j] = Config.H @ X[k][i][j] + v * a
                else:
                    ui = np.expand_dims(Config.U[i][k-1], axis=0)
                    w = np.random.normal(0, np.sqrt(Config.Q), (Config.p, 1))
                    X[k][i][j] = F @ X[k-1][i][j] + psi @ ui + Config.G @ w * a
                    v = np.random.normal(0, np.sqrt(Config.R), (Config.m, 1))
                    Y[k][i][j] = Config.H @ X[k][i][j] + v * a
    
    return Y


def critIdent(theta, params):
    """Вычисление критерия идентификации"""
    y = params['y']
    F = Config.get_F(theta)
    psi = Config.get_Psi(theta)
    Pkk = Config.P0.copy()
    
    x = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] 
          for i in range(Config.q)] for _ in range(Config.N + 1)]
    
    hi = Config.N * Config.m * Config.v * math.log(2 * math.pi)
    
    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.G, Config.Q)
        Bk1 = ABAt(Config.H, Pk1k) + Config.R
        Kk1 = Pk1k @ Config.H.T @ np.linalg.inv(Bk1)
        Pk1k1 = (np.eye(Config.n) - Kk1 @ Config.H) @ Pk1k
        
        delta = 0
        for i in range(Config.q):
            ui = np.expand_dims(Config.U[i][k], axis=0)
            for j in range(Config.ki[i]):
                if k == 0:
                    x[k][i][j] = np.zeros((Config.n, 1))
                
                xk1k = F @ x[k][i][j] + psi @ ui
                epsk1 = y[k+1][i][j] - Config.H @ xk1k
                x[k+1][i][j] = xk1k + Kk1 @ epsk1
                
                delta += (epsk1.T @ np.linalg.inv(Bk1) @ epsk1)[0, 0]
        
        hi += Config.v * math.log(np.linalg.det(Bk1)) + delta
        Pkk = Pk1k1.copy()
    
    return hi / 2.0


def grad(theta, params):
    """Аналитическое вычисление градиента критерия идентификации"""
    y = params['y']
    F = Config.get_F(theta)
    psi = Config.get_Psi(theta)
    Pkk = Config.P0.copy()
    dPkk = [mat.copy() for mat in Config.dP0]
    
    x = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] 
          for i in range(Config.q)] for _ in range(Config.N + 1)]
    
    dx = [[[[np.zeros((Config.n, 1)) for _ in range(Config.s)] 
            for _ in range(Config.ki[i])] 
           for i in range(Config.q)] for _ in range(Config.N + 1)]
    
    dhi = np.zeros(Config.s)
    
    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.G, Config.Q)
        Bk1 = ABAt(Config.H, Pk1k) + Config.R
        Kk1 = Pk1k @ Config.H.T @ np.linalg.inv(Bk1)
        Pk1k1 = (np.eye(Config.n) - Kk1 @ Config.H) @ Pk1k
        
        dBk1 = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
        dKk1 = [np.zeros((Config.n, Config.m)) for _ in range(Config.s)]
        dPk1k1 = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]
        
        for th in range(Config.s):
            dPk1k = (Config.dF[th] @ Pkk @ F.T + 
                    F @ dPkk[th] @ F.T + 
                    F @ Pkk @ Config.dF[th].T +
                    ABAt(Config.G, Config.dQ[th]) +
                    Config.dG[th] @ Config.Q @ Config.G.T +
                    Config.G @ Config.Q @ Config.dG[th].T)
            
            dBk1[th] = (Config.dH[th] @ Pk1k @ Config.H.T + 
                       Config.H @ dPk1k @ Config.H.T +
                       Config.H @ Pk1k @ Config.dH[th].T + 
                       Config.dR[th])
            
            dKk1[th] = (dPk1k @ Config.H.T + Pk1k @ Config.dH[th].T -
                       Pk1k @ Config.H.T @ np.linalg.inv(Bk1) @ dBk1[th])
            dKk1[th] = dKk1[th] @ np.linalg.inv(Bk1)
            
            dPk1k1[th] = ((np.eye(Config.n) - Kk1 @ Config.H) @ dPk1k -
                         (dKk1[th] @ Config.H + Kk1 @ Config.dH[th]) @ Pk1k)
        
        delta = np.zeros(Config.s)
        for i in range(Config.q):
            ui = np.expand_dims(Config.U[i][k], axis=0)
            for j in range(Config.ki[i]):
                if k == 0:
                    for th in range(Config.s):
                        dx[k][i][j][th] = Config.dx0[th].copy()
                    x[k][i][j] = Config.x0.copy()
                
                xk1k = F @ x[k][i][j] + psi @ ui
                epsk1 = y[k+1][i][j] - Config.H @ xk1k
                x[k+1][i][j] = xk1k + Kk1 @ epsk1
                
                for th in range(Config.s):
                    dxk1k = (Config.dF[th] @ x[k][i][j] + 
                            F @ dx[k][i][j][th] + 
                            Config.dPsi[th] @ ui)
                    
                    depsk1 = -Config.dH[th] @ xk1k - Config.H @ dxk1k
                    
                    dx[k+1][i][j][th] = (dxk1k + dKk1[th] @ epsk1 + 
                                        Kk1 @ depsk1)
                    
                    delta[th] += (depsk1.T @ np.linalg.inv(Bk1) @ epsk1)[0, 0]
                    delta[th] -= (epsk1.T @ np.linalg.inv(Bk1) @ 
                                 dBk1[th] @ np.linalg.inv(Bk1) @ epsk1)[0, 0]
        
        for th in range(Config.s):
            dhi[th] += (Config.v / 2.0) * np.trace(np.linalg.inv(Bk1) @ dBk1[th]) + delta[th]
        
        Pkk = Pk1k1.copy()
        dPkk = dPk1k1.copy()
    
    return np.array(dhi, dtype=float)


def graph(y, y2, N):
    """Построение графиков откликов"""
    x = np.arange(0, N)
    fig = plt.figure(figsize=(10, 6))
    
    plt.yscale('symlog', linthresh=1.0)
    plt.xlabel('k (Шаг)', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.plot(x, y, color='red', linewidth=3, label='Y_cp (усредненное с шумом)')
    plt.plot(x, y2, color='blue', linestyle='--', linewidth=2, label='Y* (без шума)')
    plt.title('Графики откликов системы', fontsize=14)
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    thetaMean = np.zeros(Config.s)
    yMean = np.zeros((Config.N, Config.q))
    yk1 = np.zeros((Config.N, Config.q))
    
    num_experiments = 5
    
    print("=" * 60)
    print("ИДЕНТИФИКАЦИЯ ПАРАМЕТРОВ ДИНАМИЧЕСКОЙ СИСТЕМЫ")
    print("=" * 60)
    print(f"Истинные параметры θ*: {Config.theta_true}")
    print(f"Границы поиска: θ₁ ∈ [{Config.bounds.lb[0]}, {Config.bounds.ub[0]}], "
          f"θ₂ ∈ [{Config.bounds.lb[1]}, {Config.bounds.ub[1]}]")
    print(f"Количество экспериментов: {num_experiments}")
    print("=" * 60)
    
    for i in range(num_experiments):
        Y = initY(Config.theta_true, 1)
        
        theta1 = np.random.uniform(Config.bounds.lb[0], Config.bounds.ub[0])
        theta2 = np.random.uniform(Config.bounds.lb[1], Config.bounds.ub[1])
        theta_start = np.array([theta1, theta2])
        
        result = minimize(
            fun=critIdent,
            jac=grad,
            x0=theta_start,
            args={'y': Y},
            bounds=Config.bounds,
            tol=1e-10,
            options={'maxiter': 500, 'disp': False}
        )
        
        print(f"Эксперимент {i+1}: θ_est = [{result['x'][0]:.4f}, {result['x'][1]:.4f}], "
              f"успех: {result['success']}")
        
        thetaMean += result['x']
        
        for k in range(1, Config.N):
            for j in range(Config.q):
                yMean[k, j] += np.mean(Y[k][j])
    
    thetaMean /= num_experiments
    yMean /= num_experiments
    
    sigma1 = np.linalg.norm(Config.theta_true - thetaMean) / np.linalg.norm(Config.theta_true)
    
    Y_ideal = initY(thetaMean, 0)
    for k in range(1, Config.N):
        for j in range(Config.q):
            yk1[k, j] += np.mean(Y_ideal[k][j])
    
    sigma2 = np.linalg.norm(yMean - yk1) / np.linalg.norm(yMean)
    
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ ИДЕНТИФИКАЦИИ")
    print("=" * 60)
    print(f"Средняя оценка θ (theta_cp): [{thetaMean[0]:.6f}, {thetaMean[1]:.6f}]")
    print(f"Истинные значения θ*:        [{Config.theta_true[0]:.6f}, {Config.theta_true[1]:.6f}]")
    print("-" * 60)
    print("Ошибки оценивания:")
    print(f"  В пространстве параметров (σ₁) = {sigma1:.6f} ({sigma1*100:.2f}%)")
    print(f"  В пространстве откликов (σ₂)   = {sigma2:.6f} ({sigma2*100:.2f}%)")
    print("=" * 60)
    
    graph(yMean, yk1, Config.N)


if __name__ == "__main__":
    main()