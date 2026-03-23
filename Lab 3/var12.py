import numpy as np

def compute_fim_derivatives_v12():
    # --- Исходные данные для Варианта 12 (I уровень сложности) ---
    N = 30  # Длина выборки 
    n = 2   # Размерность вектора состояния
    s = 2   # Количество неизвестных параметров
    
    # Истинные значения параметров для 12 варианта
    theta1 = -15.0
    theta2 = 0.4
    
    # Матрицы модели состояния и измерения
    F = np.array([[theta1, 1.0], 
                  [-15.0,  0.0]])
    
    dF1 = np.array([[1.0, 0.0],   # dF / d(theta_1)
                    [0.0, 0.0]])
    
    dF2 = np.array([[0.0, 0.0],   # dF / d(theta_2)
                    [0.0, 0.0]])
                    
    Psi = np.array([[5.0], 
                    [0.0]])
    
    dPsi1 = np.zeros((2, 1))      # dPsi / d(theta_1)
    dPsi2 = np.zeros((2, 1))      # dPsi / d(theta_2)
                      
    H = np.array([[1.0, 0.0]])    # dH / dtheta_i = 0
    
    Q = 0.3                       # Ковариация шума системы
    R = theta2                    # Ковариация шума измерения зависит от theta2!
    
    dR_dtheta1 = 0.0
    dR_dtheta2 = 1.0              # Производная R по theta2 равна 1
    
    Gamma = np.array([[1.0], 
                      [1.0]])
                      
    x0 = np.zeros((2, 1))
    P0 = np.array([[0.2, 0.0],    # Начальная матрица P_0
                   [0.0, 0.2]])
                   
    # Входной сигнал U^T = [5, ..., 5]
    U = np.ones((N, 1)) * 5.0
    
    # Матрица Psi_A по формуле (24)
    Psi_A = np.vstack((Psi, dPsi1, dPsi2))
    
    # --- Инициализация переменных Фильтра Калмана ---
    P_kk = P0.copy()
    dP_kk_1 = np.zeros((2, 2))
    dP_kk_2 = np.zeros((2, 2))
    K_k = np.zeros((2, 1))
    
    # --- Массивы для хранения результатов ---
    x_A = np.zeros((N + 1, n * (s + 1), 1))
    dx_A = np.zeros((N + 1, N, n * (s + 1), 1))
    dM = np.zeros((N, s, s))
    
    # --- Основной цикл по времени k ---
    for k in range(N):
        # 1. Формирование матриц F_A(t_k) (форм. 23) и вычисление x_A(t_k+1) (форм. 21)
        if k == 0:
            x_A[1] = Psi_A * U[0, 0]  # т.к. x0 = 0
            F_A = np.zeros((6, 6))
        else:
            K_tilde = F @ K_k         # K_tilde(t_k) (форм. 26)
            F_minus_KH = F - K_tilde @ H
            # Собираем блочную матрицу F_A(t_k) (учитывая, что dH/dtheta_i = 0)
            F_A = np.block([
                [F,   np.zeros((2, 2)), np.zeros((2, 2))],
                [dF1, F_minus_KH,       np.zeros((2, 2))],
                [dF2, np.zeros((2, 2)), F_minus_KH      ]
            ])
            x_A[k+1] = F_A @ x_A[k] + Psi_A * U[k, 0]
            
        # 2. Уравнения Фильтра Калмана и их производные (шаг 8)
        # Прогноз
        P_k1_k = F @ P_kk @ F.T + Gamma * Q * Gamma.T
        dP_k1_k_1 = dF1 @ P_kk @ F.T + F @ dP_kk_1 @ F.T + F @ P_kk @ dF1.T
        dP_k1_k_2 = dF2 @ P_kk @ F.T + F @ dP_kk_2 @ F.T + F @ P_kk @ dF2.T
        
        # Инновация (с учетом производных R)
        B_k1 = H @ P_k1_k @ H.T + R
        dB_k1_1 = H @ dP_k1_k_1 @ H.T + dR_dtheta1
        dB_k1_2 = H @ dP_k1_k_2 @ H.T + dR_dtheta2  # Учитываем dR/dtheta_2 = 1.0
        
        B_inv = np.linalg.inv(B_k1)
        
        # Коэффициент усиления
        K_k1 = P_k1_k @ H.T @ B_inv
        dK_k1_1 = dP_k1_k_1 @ H.T @ B_inv - P_k1_k @ H.T @ B_inv @ dB_k1_1 @ B_inv
        dK_k1_2 = dP_k1_k_2 @ H.T @ B_inv - P_k1_k @ H.T @ B_inv @ dB_k1_2 @ B_inv
        
        # Коррекция
        I_KH = np.eye(2) - K_k1 @ H
        P_k1_k1 = I_KH @ P_k1_k
        dP_k1_k1_1 = -dK_k1_1 @ H @ P_k1_k + I_KH @ dP_k1_k_1
        dP_k1_k1_2 = -dK_k1_2 @ H @ P_k1_k + I_KH @ dP_k1_k_2
        
        # 3. Вычисление производных x_A по u(t_beta) и приращений к ИМФ
        for beta in range(N):
            if beta > k:
                continue  # В силу причинности система не реагирует на будущие входы
                
            # Формулы (32), (33)
            if k == 0:
                if beta == 0:
                    dx_A[1][beta] = Psi_A
            else:
                term = Psi_A if beta == k else np.zeros((n * (s + 1), 1))
                dx_A[k+1][beta] = F_A @ dx_A[k][beta] + term
                
            # Формула (31). Т.к. dH/dtheta = 0, первые 3 слагаемых под следом равны 0.
            for i in range(s):
                for j in range(s):
                    # Вырезаем соответствующие блоки из расширенных векторов состояния
                    xi = x_A[k+1][(i+1)*2 : (i+1)*2+2]
                    dxi = dx_A[k+1][beta][(i+1)*2 : (i+1)*2+2]
                    
                    xj = x_A[k+1][(j+1)*2 : (j+1)*2+2]
                    dxj = dx_A[k+1][beta][(j+1)*2 : (j+1)*2+2]
                    
                    # Матрица внутри следа для i и j
                    inner_matrix = dxi @ xj.T + xi @ dxj.T
                    
                    # Приращение (след от матрицы 1x1 есть сам элемент)
                    inc = B_inv[0, 0] * (H @ inner_matrix @ H.T)[0, 0]
                    dM[beta, i, j] += inc
                    
        # Подготовка к следующей итерации k -> k+1
        P_kk = P_k1_k1
        dP_kk_1 = dP_k1_k1_1
        dP_kk_2 = dP_k1_k1_2
        K_k = K_k1
        
    return dM

if __name__ == "__main__":
    dM_v12 = compute_fim_derivatives_v12()
    
    print("Производные ИМФ по компонентам входного сигнала (Вариант 12, I уровень сложности)\n")
    
    print(f"Матрица dM / du(t_0):")
    print(np.round(dM_v12[0], 6))
    print("-" * 30)
    
    print(f"Матрица dM / du(t_1):")
    print(np.round(dM_v12[1], 6))
    print("-" * 30)
    
    print(f"Матрица dM / du(t_2):")
    print(np.round(dM_v12[2], 6))
    print("-" * 30)