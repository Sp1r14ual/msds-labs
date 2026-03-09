import numpy as np
import math


# ----------------------------------------------------------------------------
# задаём значение градиентов
def initValuesG():
    # ----- градиенты: шаг 1
    dF = [
        np.array([[-1, 0],
                  [0, 0]]),
        np.zeros((2, 2))
    ]

    dPsi = [np.array([[0], [0]]), np.array([[0], [1]])]
    dG = [np.array([[0], [0]]), np.array([[0], [0]])]
    dH = [np.array([[0, 0]]), np.array([[0, 0]])]
    dQ = [0, 0]
    dR = [0, 0]
    dx0 = [np.array([[0], [0]]), np.array([[0], [0]])]
    dP0 = [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])]

    return dF, dPsi, dG, dH, dQ, dR, dx0, dP0


# ----------------------------------------------------------------------------
# задаём начальные значения системы
def initValues(number, theta):
    q = 1  # кол-во точек в плане (кол-во Ui)
    N = 10  # размер вектора Ui
    n = 2  # размер вектора x
    m = 1  # размер вектора y
    r = 1  # размер вектора u
    s = 2  # количество неизвестных параметров
    p = 1  # размер вектора w, кол-во столбцов в Q

    thetaTrue = np.array([7.6, 39.5])  # истинные значения параметров

    if number == 4:
        return thetaTrue, N, q

    F =  np.array([[-theta[0], 1],[-39.5, 0]])     # матрица процесса системы (матрица перехода)
    psi =  np.array([[0], [theta[1]]])         # матрица управления
    G =  np.array([[1], [1]])                  # матрица шума
    P0 = np.array([[0.5,0],[0,0.5]])           # начальная ковариационная матрица состояния системы
    Q =  np.array([[0.2]])                     # матрица ков-ии шума процесса
    #Q = np.array([[2]])
    R = np.array([[0.5]])                      # матрица ковариации шума наблюдений
    H = np.array([[1,0]])
    x0 = np.zeros((n,1))                       #начальное состояние

    dop = np.ones(r)
    dop = dop * 5

    U = [[dop.copy() for i in range(N)] for _ in range(q)]
    ki = [1]  # кол-во наблюдений в Ui

    v = sum(ki)

    if number == 1:
        return F, psi, Q, G, R, H, U, ki, N, q, n, m, x0, p

    if number == 2:
        return P0, N, m, q, n, ki, U, F, H, R, G, Q, psi, v, x0

    if number == 3:
        return U, ki, s, F, psi, G, H, Q, R, P0, N, m, v, n, q, x0

    if number == 5:
        return F, psi, G, H, Q, R, x0, P0, s, n, r, N, U, m


# ----------------------------------------------------------------------------
# A*B*A^T, где A,B - матрицы
def ABAt(A, B):
    res = np.dot(np.dot(A, B), A.transpose())
    return res


# ----------------------------------------------------------------------------
# нахождение матрицы Ci
def cI(i, s, n):
    res = np.zeros((n, n * (s + 1)))

    for k in range(n):
        res[k, k + n * i] = 1

    return res


def dIMF(theta):
    # ----- шаг 1
    F, psi, G, H, Q, R, x0, P0, s, n, r, N, U, m = initValues(5, theta)
    dF, dPsi, dG, dH, dQ, dR, dx0, dP0 = initValuesG()  # dP0?

    # формула 24

    psiA = np.zeros((n * (s + 1), 1))

    for i in range(n):
        psiA[i][0] = psi[i][0]

    for i in range(s):
        for j in range(n):
            #psiA[i * n + n + j][0] = dPsi[i][j]
            psiA[i * n + n + j][0] = dPsi[i][j][0]

    # ----- шаг 2

    dMtheta = [[np.zeros((s, s)) for _ in range(N)] for _ in range(r)]

    Pkk = P0.copy()
    Kk = np.zeros((n, m))  # (K на шаге k)
    dKk = [np.zeros((n, m)) for _ in range(s)]  # (dK на шаге k)

    xA = np.zeros((n * (s + 1), 1))
    xAk1 = np.zeros((n * (s + 1), 1))

    dxAk = [[[np.zeros((n * (s + 1), 1)) for _ in range(N)] for _ in range(N)] for _ in range(r)]

    for k in range(N):

        # ----- шаг 3
        uk = np.expand_dims(U[0][k], axis=0)

        # ----- шаг 4
        if k == 0:

            # формула 21 при k==0

            xAk1 = np.zeros((n * (s + 1), 1))
            xAk1Temp = np.dot(F, x0) + np.dot(psi, uk)

            for i in range(n):
                xAk1[i][0] = xAk1Temp[i][0]

            for i in range(s):
                xAk1Temp = np.dot(dF[i], x0) + np.dot(F, dx0[i]) + np.dot(dPsi[i], uk)
                for j in range(n):
                    xAk1[i * n + n + j][0] = xAk1Temp[j][0]
        else:

            # ----- шаг 5

            # формула 26
            KTk = np.dot(F, Kk)

            dKTk = [np.zeros((n, m)) for _ in range(s)]

            for i in range(s):
                dKTk[i] = np.dot(dF[i], Kk) + np.dot(F, dKk[i])

            # ----- шаг 6

            # формула 23
            Fatk = np.zeros((n * (s + 1), n * (s + 1)))

            for i in range(n):
                for j in range(n):
                    Fatk[i][j] = F[i][j]

            # первый столбец
            for step in range(s):
                FatkTemp = dF[step] - np.dot(KTk, dH[step])
                for i in range(n):
                    for j in range(n):
                        Fatk[step * n + n + i][j] = FatkTemp[i][j]

            # диагонали
            FatkTemp = F - np.dot(KTk, H)
            for step in range(s):
                for i in range(n):
                    for j in range(n):
                        Fatk[step * n + n + i][step * n + n + j] = FatkTemp[i][j]

            # формула 21 при k!=0
            xAk1 = np.dot(Fatk, xA) + np.dot(psiA, uk)

        # ----- шаг 8

        # формула 10
        Pk1k = ABAt(F, Pkk) + ABAt(G, Q)
        # формула 12 (B на шаге k+1)
        Bk1 = ABAt(H, Pk1k) + R
        # формула 13 (K на шаге k+1)
        Kk1 = np.dot(np.dot(Pk1k, H.transpose()), np.linalg.inv(Bk1))
        # формула 15
        Pk1k1 = np.dot((np.eye(n) - np.dot(Kk1, H)), Pk1k)

        psiAdu = np.zeros((n * (s + 1), 1))

        # ----- шаг 9
        for b in range(N):

            # ----- шаг 10-12

            for a in range(r):

                if b != k:
                    psiAdu = np.zeros((n * (s + 1), 1))
                else:
                    dua = np.zeros((r, 1))
                    dua[a] = 1
                    psiAdu = np.dot(psiA, dua)

                if k == 0:
                    dxAk[a][k][b] = psiAdu.copy()
                else:
                    dxAk[a][k][b] = np.dot(Fatk, dxAk[a][k - 1][b]) + psiAdu

            # ----- шаг 13

            for a in range(r):

                # fl = np.expand_dims(dxAk[a][k][b], axis=0).transpose()
                h1 = np.dot(dxAk[a][k][b], xAk1.transpose()) + np.dot(xAk1, dxAk[a][k][b].transpose())

                c0 = cI(0, s, n)
                c0T = c0.T
                invB = np.linalg.inv(Bk1)

                # формула 20
                for i in range(s):
                    for j in range(s):
                        h2 = np.dot(dH[i], c0)
                        h2 = np.dot(h2, h1)
                        h2 = np.dot(h2, c0T)
                        h2 = np.dot(h2, dH[j].T)
                        h2 = np.dot(h2, invB)

                        h3 = np.dot(dH[i], c0)
                        h3 = np.dot(h3, h1)
                        h3 = np.dot(h3, (cI(j + 1, s, n)).T)
                        h3 = np.dot(h3, H.T)
                        h3 = np.dot(h3, invB)

                        h4 = np.dot(H, cI(i + 1, s, n))
                        h4 = np.dot(h4, h1)
                        h4 = np.dot(h4, c0T)
                        h4 = np.dot(h4, dH[j].T)
                        h4 = np.dot(h4, invB)

                        h5 = np.dot(H, cI(i + 1, s, n))
                        h5 = np.dot(h5, h1)
                        h5 = np.dot(h5, (cI(j + 1, s, n)).T)
                        h5 = np.dot(h5, H.T)
                        h5 = np.dot(h5, invB)

                        dMtheta[a][b][i][j] += h2[0][0] + h3[0][0] + h4[0][0] + h5[0][0]

        Kk = Kk1.copy()
        Pkk = Pk1k1.copy()
        xA = xAk1.copy()

    for a in range(r):
        for i in range(N):
            print('a = ', a)
            print('k = ', i)
            print(dMtheta[a][i])
            print()

    return (dMtheta)


dIMF(np.array([7.6, 39.5]))
