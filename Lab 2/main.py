import numpy as np
import math
    
# ----------------------------------------------------------------------------
# задаём значение градиентов
def initValuesG():
    
    # ----- градиенты: шаг 1
    #dF = [np.array([[-1, 0],[0,0]]), np.array([[0, 0],[0,0]])]
    dF = [
        np.array([[-1, 0],
                  [0, 0]]),
        np.zeros((2, 2))
    ]

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
    
    thetaTrue = np.array([7.6, 39.5])     # истинные значения параметров
    
    if number == 4:
        return thetaTrue, N, q
    
    F =  np.array([[-theta[0], 1],[-39.5, 0]])     # матрица процесса системы (матрица перехода)
    psi =  np.array([[0], [theta[1]]])         # матрица управления
    G =  np.array([[1], [1]])                  # матрица шума
    P0 = np.array([[0.5,0],[0,0.5]])           # начальная ковариационная матрица состояния системы
    Q =  np.array([[0.2]])                     # матрица ков-ии шума процесса
    R = np.array([[0.5]])                      # матрица ковариации шума наблюдений
    H = np.array([[1,0]])
    x0 = np.zeros((n,1))                       #начальное состояние
    
    dop = np.ones(r)
    dop = dop*5
    
    U = [[ dop.copy() for i in range(N)] for _ in range(q)]
    ki = [1] # кол-во наблюдений в Ui
    
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
# перемножение сложных матриц
def dotMatrix(rowA, colA, colB, n, m, A, B):
    
    res = [[ np.zeros((n,m)) for _ in range(colB)] for _ in range(rowA)]
    
    for i in range(rowA):
        for j in range(colB):
            for k in range(colA):
                pp = B[k][j]
                pp1  = A[i][k]
                res[i][j] += np.dot(A[i][k],B[k][j])
    
    
    return res

# ----------------------------------------------------------------------------
# транспонирование сложной матрицы
def transpMatrix(rowA, colA, n, m, A):
    
    res =  [[ np.zeros((m,n)) for _ in range(rowA)] for _ in range(colA)]
    
    for i in range(rowA):
        for j in range(colA):
            res[j][i] = A[i][j].T
            
    return res
    
# ----------------------------------------------------------------------------
# A*B*A^T, где A,B - матрицы
def ABAt(A,B):
    res = np.dot(np.dot(A,B),A.transpose())
    return res   

# ----------------------------------------------------------------------------
# нахождение матрицы Ci
def cI(i, s, n):
    
    res = np.zeros((n, n*(s+1)))
    
    for k in range(n):
        res[k, k+n*i] = 1

    return res

# ----------------------------------------------------------------------------
# поиск информационной матрицы Фишера
def IMF(theta):

    # ----- шаг 1
    F, psi, G, H, Q, R, x0, P0, s, n, r, N, U, m = initValues(5, theta) 
    dF, dPsi, dG, dH, dQ, dR, dx0, dP0 = initValuesG()
    

    psiA = np.zeros((n*(s+1),1))
    
    for i in range(n):
        psiA[i][0] = psi[i][0]
    
    
    for i in range(s):
        for j in range(n):
            psiA[i*n+n+j, 0] = dPsi[i][j, 0]
    
    # ----- шаг 2 
    Mtheta = np.zeros((s,s))
    
    Pkk = P0.copy()
    dPkk = dP0.copy()
    Bk = np.zeros((m,m)) # (B на шаге k)
    Kk = np.zeros((n,m)) # (K на шаге k)
    dKk = [ np.zeros((n,m)) for _ in range(s)] # (dK на шаге k)
    
    
    xA = np.zeros((n*(s+1),1))
    xAk1 = np.zeros((n*(s+1),1))
    eAk = np.zeros((n*(s+1),n*(s+1)))
    eAk1 = np.zeros((n*(s+1),n*(s+1)))

    for k in range(N):
        
        # шаг 3
        uk = np.expand_dims(U[0][k], axis=0)
        
        # ----- шаг 4
        if k!=0: 
            
            # ----- шаг 5

            # формула 26
            KTk = np.dot(F, Kk)
            
            dKTk = [ np.zeros((n,m)) for _ in range(s)]
            
            for i in range(s):
                dKTk[i] = np.dot(dF[i],Kk)+ np.dot(F, dKk[i])
            
            # ----- шаг 6
            
            # формула 23
            Fatk = np.zeros((n*(s+1),n*(s+1)))
            
            for i in range(n):
                for j in range(n):
                    Fatk[i][j] = F[i][j]
            
            # первый столбец
            for step in range(s):
                FatkTemp = dF[step] - np.dot(KTk, dH[step])
                for i in range(n):
                    for j in range(n):
                        Fatk[step*n+n+i][j] = FatkTemp[i][j]
            
            # диагонали
            FatkTemp = F - np.dot(KTk,H)
            for step in range(s):
                for i in range(n):
                    for j in range(n):
                        Fatk[step*n+n+i][step*n+n+j] = FatkTemp[i][j]
            
            
            # формула 25
            
            KAtk = np.zeros((n*(s+1),1))
            
            for i in range(n):
                KAtk[i][0] = KTk[i][0]
            
            for i in range(s):
                for j in range(n):
                    KAtk[i*n+n+j][0] = dKTk[i][j, 0]

            # ----- шаг 7
            
            # формула 21 при k!=0
            xAk1 = np.dot(Fatk, xA) + np.dot(psiA, uk)
            
            # формула 22 при k!=0
            eAk1 = ABAt(Fatk, eAk) + ABAt(KAtk, Bk)
                    
            
        else:
            # формула 21 при k==0
            
            xAk1 = np.zeros((n*(s+1),1))
            xAk1Temp = np.dot(F,x0)+np.dot(psi,uk)
            
            for i in range(n):
                xAk1[i][0] = xAk1Temp[i][0]
            
            for i in range(s):
                xAk1Temp = np.dot(dF[i],x0)+np.dot(F,dx0[i])+np.dot(dPsi[i],uk)
                for j in range(n):
                    xAk1[i*n+n+j][0] = xAk1Temp[j][0]
            
            # формула 22 при k==0
            eAk1 = np.zeros((n*(s+1),n*(s+1)))
        
        # ----- шаг 8
        
        # ---------------------------------------------------------------------
        # вычисления, совпадающие с алгоритмами в лр 1
        
        
        # формула 10 
        Pk1k = ABAt(F,Pkk) + ABAt(G,Q)
        # формула 12 (B на шаге k+1)
        Bk1 = ABAt(H,Pk1k) + R
        # формула 13 (K на шаге k+1)
        Kk1 = np.dot(np.dot(Pk1k, H.transpose()), np.linalg.inv(Bk1))
        # формула 15 
        Pk1k1 = np.dot((np.eye(n)-np.dot(Kk1,H)),Pk1k)
        
        # ----- шаг 4 (лр 1, поиск градиента)
        dBk1 = [ np.zeros((m,m)) for _ in range(s)]
        dKk1 = [ np.zeros((n,m)) for _ in range(s)]
        dPk1k1 = [ np.zeros((n,n)) for _ in range(s)]
        
        for th in range(s):
            
            # +
            
            dPk1k = np.dot(np.dot(dF[th], Pkk),F.transpose())
            dPk1k += ABAt(F,dPkk[th]) + ABAt(G,dQ[th])
            dPk1k += np.dot(np.dot(F,Pkk),dF[th].transpose())
            dPk1k += np.dot(np.dot(dG[th],Q),G.transpose())
            dPk1k += np.dot(np.dot(G,Q), dG[th].transpose())
            
            # +
            dBk1[th] = np.dot((np.dot(dH[th], Pk1k)), H.transpose())
            dBk1[th] = dBk1[th] + ABAt(H, dPk1k)
            dBk1[th] = dBk1[th] + np.dot(np.dot(H,Pk1k),dH[th].transpose()) + dR[th]
            
            # +
            dKk1[th] = np.dot(dPk1k,H.transpose()) + np.dot(Pk1k,dH[th].transpose())
            dKk1[th] = dKk1[th] - np.dot(np.dot(np.dot(Pk1k,H.transpose()), np.linalg.inv(Bk1)), dBk1[th])
            dKk1[th] = np.dot(dKk1[th],np.linalg.inv(Bk1))
            
            # + 
            dPk1k1[th] = np.dot(np.eye(n)-np.dot(Kk1,H), dPk1k)
            dPk1k1[th] -= np.dot((np.dot(dKk1[th],H)+np.dot(Kk1,dH[th])), Pk1k)   
  
        
        # ---------------------------------------------------------------------
        
        # ----- шаг 9
        h1 = np.dot(xAk1, xAk1.T) + eAk1
                
        c0 = cI(0, s, n)
        c0T = c0.T
        invB = np.linalg.inv(Bk1)
        
        # формула 20
        for i in range(s):
            for j in range(s):
                
                h2 = np.dot(dH[i], c0)
                h2 = np.dot(h2,h1)
                h2 = np.dot(h2,c0T)
                h2 = np.dot(h2,dH[j].T)
                h2 = np.dot(h2, invB)
                
                h3 = np.dot(dH[i], c0)
                h3 = np.dot(h3,h1)
                h3 = np.dot(h3,(cI(j+1, s, n)).T)
                h3 = np.dot(h3,H.T)
                h3 = np.dot(h3, invB)
                
                h4 = np.dot(H, cI(i+1, s, n))
                h4 = np.dot(h4,h1)
                h4 = np.dot(h4,c0T)
                h4 = np.dot(h4,dH[j].T)
                h4 = np.dot(h4, invB)
                
                h5 = np.dot(H, cI(i+1, s, n))
                h5 = np.dot(h5,h1)
                h5 = np.dot(h5,(cI(j+1, s, n)).T)
                h5 = np.dot(h5,H.T)
                h5 = np.dot(h5, invB)
                
                h6 = np.dot(dBk1[i], np.linalg.inv(Bk1))
                h6 = np.dot(h6, dBk1[j])
                h6 = np.dot(h6, np.linalg.inv(Bk1))
                h6 = 0.5*h6
                
                Mtheta[i][j] += h2[0][0] + h3[0][0] + h4[0][0] + h5[0][0] + h6[0][0]
             
        Bk = Bk1.copy()
        Kk = Kk1.copy()
        Pkk = Pk1k1.copy()
        dKk = dKk1.copy()
        dPkk = dPk1k1.copy()
        xA = xAk1.copy()
        eAk = eAk1.copy()
                    
    print(Mtheta)
    print("Symmetric:", np.allclose(Mtheta, Mtheta.T))
    print("Eigenvalues:", np.linalg.eigvals(Mtheta))
        

IMF(np.array([7.6, 39.5]))

