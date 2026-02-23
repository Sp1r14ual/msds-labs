import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds

# ----------------------------------------------------------------------------
# задаём значение градиентов
def initValuesG():

    # ----- градиенты для Варианта 5: шаг 1
    # Матрица F = [[-theta1, 1], [-39.5, 0]]
    dF = [np.array([[-1, 0],[0,0]]), np.array([[0, 0],[0,0]])]
    
    # Матрица Psi = [[0], [theta2]]
    dPsi = [np.array([[0], [0]]), np.array([[0], [1]])]
    
    dG = [np.array([[0], [0]]), np.array([[0], [0]])]
    dH = [np.array([[0,0]]), np.array([[0,0]])]
    dQ = [0, 0]
    dR = [0, 0]
    dx0 =  [np.array([[0], [0]]), np.array([[0], [0]])]
    dP0 = [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])]

    return dF, dPsi, dG, dH, dQ, dR, dx0, dP0

# ----------------------------------------------------------------------------
# задаём начальные значения системы
def initValues(number, theta):

    q = 1 # кол-во точек в плане (кол-во Ui)
    N = 30 # размер вектора Ui
    n = 2 # размер вектора x
    m = 1 # размер вектора y
    r = 1 # размер вектора u
    s = 2 # количество неизвестных параметров
    p = 1 # размер вектора w, кол-во столбцов в Q

    # истинные значения параметров для Варианта 5
    thetaTrue = np.array([7.6, 39.5])     

    if number == 4:
        return thetaTrue, N, q

    # Матрицы для Варианта 5
    F =  np.array([[-theta[0], 1], [-39.5, 0]])     # матрица процесса системы
    psi =  np.array([[0], [theta[1]]])              # матрица управления
    G =  np.array([[1], [1]])                       # В пособии это матрица Gamma
    P0 = np.array([[0.5, 0], [0, 0.5]])
    Q =  np.array([[0.2]])
    R =  np.array([[0.5]])
    H =  np.array([[1, 0]])
    x0 = np.zeros((n,1))

    # Входной сигнал U = 5 (для варианта 5)
    U = [[ 5.0 * np.ones(r) for i in range(N)] for _ in range(q)]
    ki = [1] # кол-во наблюдений в Ui

    v = sum(ki)

    if number == 1:
        return F, psi, Q, G, R, H, U, ki, N, q, n, m, x0, p

    if number == 2:
        return P0, N, m, q, n, ki, U, F, H, R, G, Q, psi, v, x0

    if number == 3:
        return U, ki, s, F, psi, G, H, Q, R, P0, N, m, v, n, q, x0


# ----------------------------------------------------------------------------
# задаём выход системы
def initY(theta, flag):

    F, psi, Q, G, R, H, U, ki, N, q, n, m, x0, p = initValues(1, np.array(theta))

    a = 0
    if flag == 1:
        a = 1

    # ----- начальные значения X и Y
    X = [[[ np.zeros((n,1)) for j in range(ki[i])] for i in range(q)] for _ in range(N+1)]
    Y = [[[ np.zeros((m,1)) for j in range(ki[i])] for i in range(q)] for _ in range(N+1)]

    for k in range(N+1):
        for i in range(q):
            for j in range(ki[i]):
                if k == 0:
                    X[k][i][j] = x0.copy()
                    v = np.random.normal(0, np.sqrt(R), (m,1))
                    Y[k][i][j] =  np.dot(H, X[k][i][j]) + v*a
                else:
                    ui = np.expand_dims(U[i][k-1], axis=0)

                    w = np.random.normal(0, np.sqrt(Q), (p,1))
                    X[k][i][j] = np.dot(F, X[k-1][i][j]) + np.dot(psi, ui) + np.dot(G, w)*a

                    v = np.random.normal(0, np.sqrt(R), (m,1))
                    Y[k][i][j] =  np.dot(H, X[k][i][j]) + v*a

    return Y

# ----------------------------------------------------------------------------
# A*B*A^T, где A,B - матрицы
def ABAt(A,B):
    res = np.dot(np.dot(A,B),A.transpose())
    return res

# ----------------------------------------------------------------------------
# вычисление критерия идентификации
def critIdent(theta, params):

    y = params['y']
    Pkk, N, m, q, n, ki, U, F, H, R, G, Q, psi, v, x0 = initValues(2, theta)
    x = [[[ np.zeros((n,1)) for j in range(ki[i])] for i in range(q)] for _ in range(N+1)]
    
    hi = N*m*v*math.log(2*math.pi)

    for k in range(N):
        Pk1k = ABAt(F,Pkk) + ABAt(G,Q)
        Bk1 = ABAt(H,Pk1k) + R
        Kk1 = np.dot(np.dot(Pk1k, H.transpose()), np.linalg.inv(Bk1))
        Pk1k1 = np.dot((np.eye(n)-np.dot(Kk1,H)),Pk1k)
        
        delta = 0
        for i in range(q):
            ui = np.expand_dims(U[i][k], axis=0)
            for j in range(ki[i]):
                if k == 0:
                    x[k][i][j] = np.zeros((n,1))

                xk1k = np.dot(F,x[k][i][j])+np.dot(psi,ui)
                epsk1 = y[k+1][i][j] - np.dot(H,xk1k)
                x[k+1][i][j] = xk1k+np.dot(Kk1,epsk1)

                delta = delta + np.dot( np.dot( epsk1.transpose(), np.linalg.inv(Bk1) ), epsk1)

        hi = hi + v*math.log(np.linalg.det(Bk1)) + delta
        Pkk = Pk1k1.copy()

    hi = hi/2
    res = hi[0][0]
    return res

# ----------------------------------------------------------------------------
# вычисление градиента критерия идентификации
def grad(theta, params):

    y = params['y']
    U, ki, s, F, psi, G, H, Q, R, Pkk, N, m, v, n, q, x0 = initValues(3, theta)

    x = [[[ np.zeros((n,1)) for j in range(ki[i])] for i in range(q)] for _ in range(N+1)]
    dx = [[[[ np.zeros((n,1)) for p in range(s)] for j in range(ki[i])] for i in range(q)] for _ in range(N+1)]

    dF, dPsi, dG, dH, dQ, dR, dx0, dPkk = initValuesG()
    dhi = np.zeros(s)

    for k in range(N):
        Pk1k = ABAt(F,Pkk) + ABAt(G,Q)
        Bk1 = ABAt(H,Pk1k) + R
        Kk1 = np.dot(np.dot(Pk1k, H.transpose()), np.linalg.inv(Bk1))
        Pk1k1 = np.dot((np.eye(n)-np.dot(Kk1,H)),Pk1k)

        dBk1 = [ np.zeros((m,m)) for _ in range(s)]
        dKk1 = [ np.zeros((n,m)) for _ in range(s)]
        dPk1k1 = [ np.zeros((n,n)) for _ in range(s)]

        for th in range(s):
            dPk1k = np.dot(np.dot(dF[th], Pkk),F.transpose())
            dPk1k += ABAt(F,dPkk[th]) + ABAt(G,dQ[th])
            dPk1k += np.dot(np.dot(F,Pkk),dF[th].transpose())
            dPk1k += np.dot(np.dot(dG[th],Q),G.transpose())
            dPk1k += np.dot(np.dot(G,Q), dG[th].transpose())

            dBk1[th] = np.dot((np.dot(dH[th], Pk1k)), H.transpose())
            dBk1[th] = dBk1[th] + ABAt(H, dPk1k)
            dBk1[th] = dBk1[th] + np.dot(np.dot(H,Pk1k),dH[th].transpose()) + dR[th]

            dKk1[th] = np.dot(dPk1k,H.transpose()) + np.dot(Pk1k,dH[th].transpose())
            dKk1[th] = dKk1[th] - np.dot(np.dot(np.dot(Pk1k,H.transpose()), np.linalg.inv(Bk1)), dBk1[th])
            dKk1[th] = np.dot(dKk1[th],np.linalg.inv(Bk1))

            dPk1k1[th] = np.dot(np.eye(n)-np.dot(Kk1,H), dPk1k)
            dPk1k1[th] -= np.dot((np.dot(dKk1[th],H)+np.dot(Kk1,dH[th])), Pk1k)

        delta = np.zeros(s)

        for i in range(q):
            ui = np.expand_dims(U[i][k], axis=0)
            for j in range(ki[i]):
                if k==0:
                    dx[k][i][j] = dx0.copy()
                    x[k][i][j] = x0.copy()

                xk1k = np.dot(F,x[k][i][j])+np.dot(psi,ui)
                epsk1 = y[k+1][i][j] - np.dot(H,xk1k)
                x[k+1][i][j] = xk1k+np.dot(Kk1,epsk1)

                for th in range(s):
                    depsk1 = np.zeros((m,1))
                    dxk1k = np.dot(dF[th],x[k][i][j]) + np.dot(F,dx[k][i][j][th])+np.dot(dPsi[th],ui)
                    depsk1 = depsk1 - np.dot(dH[th],xk1k) - np.dot(H,dxk1k)
                    dx[k+1][i][j][th] = dxk1k + np.dot(dKk1[th],epsk1)+np.dot(Kk1,depsk1)

                    delta[th] += (depsk1.T @ np.linalg.inv(Bk1) @ epsk1)[0, 0]
                    delta[th] -= (
                        (epsk1.T / 2 @ np.linalg.inv(Bk1) @ dBk1[th] @ np.linalg.inv(Bk1) @ epsk1)[0, 0]
                    )

        for th in range(s):
            dhi[th] += (v/2)*np.trace(np.dot(np.linalg.inv(Bk1), dBk1[th])) + delta[th]

        Pkk = Pk1k1.copy()
        dPkk = dPk1k1.copy()

    return np.array(dhi, dtype=float)

# ----------------------------------------------------------------------------
# Главная функция запуска
def lab1():

    thetaMean = np.zeros(2)
    thetaTrue, N, q = initValues(4, thetaMean)
    yMean = np.zeros((N, q))
    yk1 = np.zeros((N, q))

    for i in range(5):

        Y = initY(thetaTrue, 1)

        # Начальное приближение берется в пределах ОДЗ для варианта 5
        # theta1 in [1; 10], theta2 in [30; 45]
        theta1 = np.random.uniform(1.0, 10.0)
        theta2 = np.random.uniform(30.0, 45.0)

        theta = np.array([theta1, theta2])

        result = minimize(
            fun=critIdent,
            jac=grad,
            x0=theta,
            args={'y': Y},
            # ИСПРАВЛЕНИЕ: Bounds принимает ([нижние границы], [верхние границы])
            bounds=Bounds([1.0, 30.0], [10.0, 45.0]), 
            tol=1e-10
        )

        print(f"Эксперимент {i+1}, Оценка параметров: {result['x']}")

        thetaMean[0] += result['x'][0]
        thetaMean[1] += result['x'][1]

        for k in range(1,N):
            for j in range(q):
                yMean[k, j] += np.mean(Y[k][j])

    # Усреднение результатов по 5 экспериментам
    thetaMean[0] = thetaMean[0]/5
    thetaMean[1] = thetaMean[1]/5
    yMean = yMean / 5.0 # Исправление: необходимо поделить на количество экспериментов

    # ----- ошибки оценивания
    # В пространстве параметров
    sigma1 = np.linalg.norm(thetaTrue - thetaMean) / np.linalg.norm(thetaTrue)

    # В пространстве откликов (без шума, a=0)
    Y = initY(thetaMean, 0)
    for k in range(1,N):
        for j in range(q):
            yk1[k, j] += np.mean(Y[k][j])

    sigma2 = np.linalg.norm(yMean - yk1) / np.linalg.norm(yMean)

    print("-" * 30)
    print(f"Средняя оценка theta (theta_cp): {thetaMean}")
    print("Ошибки оценивания: ")
    print("delta_theta (sigma1) = ", sigma1)
    print("delta_Y (sigma2) = ", sigma2)

    def graph(y, y2, N):
        x = np.arange(0, N)
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot(111)

        plt.xlabel('k (Шаг)')
        plt.ylabel('Y')
        plt.plot(x, y,'red',  label='Y_cp (усредненное по запускам с шумом)')
        plt.plot(x, y2,'blue',  label='Y* (по найденным параметрам без шума)')
        plt.yscale('symlog', linthresh=1.0)
        plt.title('Графики Y и Y*')
        plt.legend()
        plt.grid(True)
        plt.show()

    def graph2(y, y2, N):
        x = np.arange(0, N)
        fig = plt.figure(figsize = (10, 6))
        ax = fig.add_subplot(111)

        plt.xlabel('k (Шаг)')
        plt.ylabel('Y (Симметричный логарифмический масштаб)')
        
        # Делаем линию Y_cp сплошной и толстой
        plt.plot(x, y, color='red', linewidth=3, label='Y_cp (усредненное по запускам с шумом)')
        
        # Делаем линию Y* пунктирной, чтобы сквозь неё было видно красную
        plt.plot(x, y2, color='white', linestyle='--', linewidth=2, label='Y* (по найденным параметрам без шума)')

        # Включаем симметричный логарифмический масштаб оси Y
        # linthresh определяет порог около нуля, где масштаб остается линейным (чтобы логарифм нуля не выдал ошибку)
        plt.yscale('symlog', linthresh=1.0)
        
        plt.title('Графики Y и Y* в логарифмическом масштабе')
        
        # Настраиваем сетку, чтобы было видно логарифмические шаги (10^1, 10^2, 10^3 и т.д.)
        plt.grid(True, which="both", ls="-", color='0.8')
        
        plt.legend()
        plt.show()

    graph(yMean, yk1, N)

if __name__ == "__main__":
    lab1()