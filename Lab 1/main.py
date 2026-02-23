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
    n = 2           # Размер вектора состояния x
    m = 1           # Размер вектора измерений y
    r = 1           # Размер вектора управления u
    p = 1           # Размер вектора шума системы w
    s = 2           # Количество неизвестных параметров theta

    # 2. ИСТИННЫЕ ЗНАЧЕНИЯ И ГРАНИЦЫ ПОИСКА
    theta_true = np.array([7.6, 39.5])
    bounds = Bounds([1.0, 30.0], [10.0, 45.0]) # [нижние границы], [верхние границы]

    # 3. ПОСТОЯННЫЕ МАТРИЦЫ И НАЧАЛЬНЫЕ УСЛОВИЯ
    G = np.array([[1], [1]])                 
    H = np.array([[1, 0]])
    Q = np.array([[0.2]])                    # Дисперсия шума системы
    R = np.array([[0.5]])                    # Дисперсия шума измерений
    P0 = np.array([[0.5, 0], [0, 0.5]])      # Начальная ковариационная матрица
    x0 = np.zeros((n, 1))                    # Начальное состояние x_0

    # 4. ФУНКЦИИ МАТРИЦ, ЗАВИСЯЩИХ ОТ ПАРАМЕТРОВ THETA
    @staticmethod
    def get_F(theta):
        return np.array([[-theta[0], 1], [-39.5, 0]])

    @staticmethod
    def get_Psi(theta):
        return np.array([[0], [theta[1]]])

    # 5. ГРАДИЕНТЫ МАТРИЦ ПО ПАРАМЕТРАМ THETA (d/d_theta1, d/d_theta2)
    # Так как зависимость линейная, матрицы градиентов постоянны.
    dF = [np.array([[-1, 0], [0, 0]]), np.array([[0, 0], [0, 0]])]
    dPsi = [np.array([[0], [0]]), np.array([[0], [1]])]

# Входной сигнал U = 5 (постоянный для всех шагов)
Config.U = [[5.0 * np.ones(Config.r) for _ in range(Config.N)] for _ in range(Config.q)]

# Градиенты для константных матриц (состоят из нулей)
Config.dG = [np.zeros((Config.n, Config.p)) for _ in range(Config.s)]
Config.dH = [np.zeros((Config.m, Config.n)) for _ in range(Config.s)]
Config.dQ = [np.zeros((Config.p, Config.p)) for _ in range(Config.s)]
Config.dR = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
Config.dx0 = [np.zeros((Config.n, 1)) for _ in range(Config.s)]
Config.dP0 = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]

def ABAt(A, B):
    """Вычисление A * B * A^T"""
    return np.dot(np.dot(A, B), A.transpose())

def initY(theta, flag):
    """Генерация выхода системы (измерений Y)"""
    F = Config.get_F(theta)
    psi = Config.get_Psi(theta)
    a = 1 if flag == 1 else 0

    X = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]
    Y = [[[np.zeros((Config.m, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]

    for k in range(Config.N + 1):
        for i in range(Config.q):
            for j in range(Config.ki[i]):
                if k == 0:
                    X[k][i][j] = Config.x0.copy()
                    v = np.random.normal(0, np.sqrt(Config.R), (Config.m, 1))
                    Y[k][i][j] = np.dot(Config.H, X[k][i][j]) + v * a
                else:
                    ui = np.expand_dims(Config.U[i][k-1], axis=0)
                    w = np.random.normal(0, np.sqrt(Config.Q), (Config.p, 1))
                    X[k][i][j] = np.dot(F, X[k-1][i][j]) + np.dot(psi, ui) + np.dot(Config.G, w) * a
                    v = np.random.normal(0, np.sqrt(Config.R), (Config.m, 1))
                    Y[k][i][j] = np.dot(Config.H, X[k][i][j]) + v * a
    return Y

def critIdent(theta, params):
    """Вычисление критерия идентификации (логарифмическая функция правдоподобия)"""
    # Шаг 1
    y = params['y']
    F = Config.get_F(theta)
    psi = Config.get_Psi(theta)
    Pkk = Config.P0.copy()

    # Шаг 2
    x = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]
    hi = Config.N * Config.m * Config.v * math.log(2 * math.pi)

    # Шаг 3
    for k in range(Config.N):
        Pk1k = ABAt(F, Pkk) + ABAt(Config.G, Config.Q) # Формула 10
        Bk1 = ABAt(Config.H, Pk1k) + Config.R # Формула 12
        Kk1 = np.dot(np.dot(Pk1k, Config.H.transpose()), np.linalg.inv(Bk1)) # Формула 13
        Pk1k1 = np.dot((np.eye(Config.n) - np.dot(Kk1, Config.H)), Pk1k) # Формула 15
        
        # Шаг 4
        delta = 0
        for i in range(Config.q):
            ui = np.expand_dims(Config.U[i][k], axis=0) # Шаг 5
            for j in range(Config.ki[i]): # Шаг 6
                if k == 0: # Шаг 7
                    x[k][i][j] = np.zeros((Config.n, 1)) 

                # Шаг 8
                xk1k = np.dot(F, x[k][i][j]) + np.dot(psi, ui) # Формула 9
                epsk1 = y[k+1][i][j] - np.dot(Config.H, xk1k) # Формула 11
                x[k+1][i][j] = xk1k + np.dot(Kk1, epsk1) # Формула 14

                # Шаг 9
                delta += np.dot(np.dot(epsk1.transpose(), np.linalg.inv(Bk1)), epsk1)[0, 0]

        hi += Config.v * math.log(np.linalg.det(Bk1)) + delta # крит идент., шаг 12
        Pkk = Pk1k1.copy()

    return hi / 2.0 # шаг 14

def grad(theta, params):
    """Аналитическое вычисление градиента критерия идентификации"""
    # Шаг 1-2
    y = params['y']
    F = Config.get_F(theta)
    psi = Config.get_Psi(theta)
    
    Pkk = Config.P0.copy()
    dPkk = [mat.copy() for mat in Config.dP0]
    
    x = [[[np.zeros((Config.n, 1)) for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]
    dx = [[[[np.zeros((Config.n, 1)) for _ in range(Config.s)] for _ in range(Config.ki[i])] for i in range(Config.q)] for _ in range(Config.N + 1)]

    dhi = np.zeros(Config.s)

    for k in range(Config.N):
        # Шаг 3
        Pk1k = ABAt(F, Pkk) + ABAt(Config.G, Config.Q)
        Bk1 = ABAt(Config.H, Pk1k) + Config.R
        Kk1 = np.dot(np.dot(Pk1k, Config.H.transpose()), np.linalg.inv(Bk1))
        Pk1k1 = np.dot((np.eye(Config.n) - np.dot(Kk1, Config.H)), Pk1k)
        
        # Шаг 4
        dBk1 = [np.zeros((Config.m, Config.m)) for _ in range(Config.s)]
        dKk1 = [np.zeros((Config.n, Config.m)) for _ in range(Config.s)]
        dPk1k1 = [np.zeros((Config.n, Config.n)) for _ in range(Config.s)]

        for th in range(Config.s):
            dPk1k = np.dot(np.dot(Config.dF[th], Pkk), F.transpose())
            dPk1k += ABAt(F, dPkk[th]) + ABAt(Config.G, Config.dQ[th])
            dPk1k += np.dot(np.dot(F, Pkk), Config.dF[th].transpose())
            dPk1k += np.dot(np.dot(Config.dG[th], Config.Q), Config.G.transpose())
            dPk1k += np.dot(np.dot(Config.G, Config.Q), Config.dG[th].transpose())

            dBk1[th] = np.dot(np.dot(Config.dH[th], Pk1k), Config.H.transpose())
            dBk1[th] += ABAt(Config.H, dPk1k)
            dBk1[th] += np.dot(np.dot(Config.H, Pk1k), Config.dH[th].transpose()) + Config.dR[th]

            dKk1[th] = np.dot(dPk1k, Config.H.transpose()) + np.dot(Pk1k, Config.dH[th].transpose())
            dKk1[th] -= np.dot(np.dot(np.dot(Pk1k, Config.H.transpose()), np.linalg.inv(Bk1)), dBk1[th])
            dKk1[th] = np.dot(dKk1[th], np.linalg.inv(Bk1))

            dPk1k1[th] = np.dot(np.eye(Config.n) - np.dot(Kk1, Config.H), dPk1k)
            dPk1k1[th] -= np.dot((np.dot(dKk1[th], Config.H) + np.dot(Kk1, Config.dH[th])), Pk1k)

        # Шаг 5
        delta = np.zeros(Config.s)

        for i in range(Config.q):
            ui = np.expand_dims(Config.U[i][k], axis=0) #Шаг 6
            # Шаг 7
            for j in range(Config.ki[i]):
                # Шаг 8
                if k == 0:
                    for th in range(Config.s):
                        dx[k][i][j][th] = Config.dx0[th].copy()
                    x[k][i][j] = Config.x0.copy()

                # Шаг 9
                xk1k = np.dot(F, x[k][i][j]) + np.dot(psi, ui)
                epsk1 = y[k+1][i][j] - np.dot(Config.H, xk1k)
                x[k+1][i][j] = xk1k + np.dot(Kk1, epsk1)

                for th in range(Config.s):
                    # Шаг 10
                    dxk1k = np.dot(Config.dF[th], x[k][i][j]) + np.dot(F, dx[k][i][j][th]) + np.dot(Config.dPsi[th], ui)
                    depsk1 = -np.dot(Config.dH[th], xk1k) - np.dot(Config.H, dxk1k)
                    dx[k+1][i][j][th] = dxk1k + np.dot(dKk1[th], epsk1) + np.dot(Kk1, depsk1)

                    # Шаг 11
                    delta[th] += (depsk1.T @ np.linalg.inv(Bk1) @ epsk1)[0, 0]
                    delta[th] -= (epsk1.T / 2 @ np.linalg.inv(Bk1) @ dBk1[th] @ np.linalg.inv(Bk1) @ epsk1)[0, 0]

        # Шаг 14
        for th in range(Config.s):
            dhi[th] += (Config.v / 2.0) * np.trace(np.dot(np.linalg.inv(Bk1), dBk1[th])) + delta[th]

        Pkk = Pk1k1.copy()
        dPkk = dPk1k1.copy()

    return np.array(dhi, dtype=float)


def main():
    thetaMean = np.zeros(Config.s)
    yMean = np.zeros((Config.N, Config.q))
    yk1 = np.zeros((Config.N, Config.q))
    
    num_experiments = 5

    for i in range(num_experiments):
        Y = initY(Config.theta_true, 1)

        # Выбираем случайное начальное приближение в пределах границ
        theta1 = np.random.uniform(Config.bounds.lb[0], Config.bounds.ub[0])
        theta2 = np.random.uniform(Config.bounds.lb[1], Config.bounds.ub[1])
        theta_start = np.array([theta1, theta2])

        result = minimize(
            fun=critIdent,
            jac=grad,
            x0=theta_start,
            args={'y': Y},
            bounds=Config.bounds, 
            tol=1e-10
        )

        print(f"Эксперимент {i+1}, Оценка параметров: {result['x']}")

        thetaMean += result['x']

        for k in range(1, Config.N):
            for j in range(Config.q):
                yMean[k, j] += np.mean(Y[k][j])

    # Усреднение результатов по всем экспериментам
    thetaMean /= num_experiments
    yMean /= num_experiments

    # Расчет ошибок оценивания
    sigma1 = np.linalg.norm(Config.theta_true - thetaMean) / np.linalg.norm(Config.theta_true)

    # Генерация откликов без шума по найденным средним параметрам
    Y_ideal = initY(thetaMean, 0)
    for k in range(1, Config.N):
        for j in range(Config.q):
            yk1[k, j] += np.mean(Y_ideal[k][j])

    sigma2 = np.linalg.norm(yMean - yk1) / np.linalg.norm(yMean)

    print("-" * 30)
    print(f"Средняя оценка theta (theta_cp): {thetaMean}")
    print("Ошибки оценивания: ")
    print(f"В пространстве параметров (sigma1) = {sigma1:.6f}")
    print(f"В пространстве откликов   (sigma2) = {sigma2:.6f}")

    # Построение графиков
    def graph(y, y2, N):
        x = np.arange(0, N)
        fig = plt.figure(figsize=(10, 6))
        
        # Логарифмический симметричный масштаб
        plt.yscale('symlog', linthresh=1.0)
        
        plt.xlabel('k (Шаг)')
        plt.ylabel('Y')
        
        plt.plot(x, y, color='red', linewidth=3, label='Y_cp (усредненное по запускам с шумом)')
        plt.plot(x, y2, color='blue', linestyle='--', linewidth=2, label='Y* (по найденным параметрам без шума)')
        
        plt.title('Графики Y и Y*')
        plt.grid(True, which="both", ls="-", color='0.8')
        plt.legend()
        plt.show()

    graph(yMean, yk1, Config.N)

if __name__ == "__main__":
    main()