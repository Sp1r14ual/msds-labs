import numpy as np

def compute_fim_derivatives():
    # --- Исходные данные для Варианта 1 (I уровень сложности) ---
    N = 30  # Длина выборки (можно изменить, обычно N=30)
    n = 2   # Размерность вектора состояния
    s = 2   # Количество неизвестных параметров
    
    # Истинные значения параметров
    theta1 = -0.8
    theta2 = 1.0
    
    # Матрицы модели состояния и измерения
    F = np.array([[theta1, 1.0], 
                  [-1.5,   0.0]])
    
    dF1 = np.array([[1.0, 0.0],   # dF / d(theta_1)
                    [0.0, 0.0]])
    
    dF2 = np.array([[0.0, 0.0],   # dF / d(theta_2)
                    [0.0, 0.0]])
                    
    Psi = np.array([[1.0], 
                    [theta2]])
    
    dPsi1 = np.array([[0.0],      # dPsi / d(theta_1)
                      [0.0]])
    
    dPsi2 = np.array([[0.0],      # dPsi / d(theta_2)
                      [1.0]])
                      
    H = np.array([[1.0, 0.0]])
    # dH / d(theta_i) = 0
    
    Q = 0.5
    R = 0.1
    Gamma = np.array([[1.0], 
                      [1.0]])
                      
    x0 = np.zeros((2, 1))
    P0 = np.array([[0.1, 0.0], 
                   [0.0, 0.1]])
                   
    # Входной сигнал U^T = [1, ..., 1]
    U = np.ones((N, 1))
    
    # Матрица Psi_A по формуле (24)
    Psi_A = np.vstack((Psi, dPsi1, dPsi2))
    
    # --- Инициализация переменных Фильтра Калмана ---
    P_kk = P0.copy()
    dP_kk_1 = np.zeros((2, 2))
    dP_kk_2 = np.zeros((2, 2))
    K_k = np.zeros((2, 1))
    
    # --- Массивы для хранения результатов ---
    # x_A[k] будет хранить расширенный вектор состояния на шаге k
    x_A = np.zeros((N + 1, n * (s + 1), 1))
    
    # dx_A[k][beta] - производная x_A(t_k) по u(t_beta)
    dx_A = np.zeros((N + 1, N, n * (s + 1), 1))
    
    # dM[beta, i, j] - производная ИМФ dM_{ij} по u(t_beta)
    dM = np.zeros((N, s, s))
    
    # --- Основной цикл по времени k ---
    for k in range(N):
        # 1. Формирование матриц F_A(t_k) (форм. 23) и вычисление x_A(t_k+1) (форм. 21)
        if k == 0:
            x_A[1] = Psi_A * U[0, 0]  # т.к. x0 = 0
            F_A = np.zeros((6, 6))    # На нулевом шаге F_A не используется для x_A[1]
        else:
            K_tilde = F @ K_k         # K_tilde(t_k) (форм. 26)
            F_minus_KH = F - K_tilde @ H
            # Собираем блочную матрицу F_A(t_k)
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
        
        # Инновация
        B_k1 = H @ P_k1_k @ H.T + R
        dB_k1_1 = H @ dP_k1_k_1 @ H.T
        dB_k1_2 = H @ dP_k1_k_2 @ H.T
        
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
            # Оставшееся слагаемое превращается в скалярную операцию с блоками C_i
            for i in range(s):
                for j in range(s):
                    # Вырезаем соответствующие блоки (размер 2х1)
                    # i+1 и j+1 используются, т.к. блок 0 - это базовое состояние (не производная)
                    xi = x_A[k+1][(i+1)*2 : (i+1)*2+2]
                    dxi = dx_A[k+1][beta][(i+1)*2 : (i+1)*2+2]
                    
                    xj = x_A[k+1][(j+1)*2 : (j+1)*2+2]
                    dxj = dx_A[k+1][beta][(j+1)*2 : (j+1)*2+2]
                    
                    # Матрица внутри следа (dx_A*x_A^T + x_A*dx_A^T) для i и j
                    inner_matrix = dxi @ xj.T + xi @ dxj.T
                    
                    # Приращение (след от матрицы размерности 1x1 есть сам элемент)
                    inc = B_inv[0, 0] * (H @ inner_matrix @ H.T)[0, 0]
                    dM[beta, i, j] += inc
                    
        # Подготовка к следующей итерации k -> k+1
        P_kk = P_k1_k1
        dP_kk_1 = dP_k1_k1_1
        dP_kk_2 = dP_k1_k1_2
        K_k = K_k1
        
    return dM

if __name__ == "__main__":
    dM = compute_fim_derivatives()
    
    # Выведем производные для нескольких компонентов входа, 
    # чтобы продемонстрировать работоспособность.
    print("Производные ИМФ по компонентам входного сигнала (Вариант 1, I уровень сложности)\n")
    
    print(f"Матрица dM / du(t_0):")
    print(np.round(dM[0], 4))
    print("-" * 30)
    
    print(f"Матрица dM / du(t_1):")
    print(np.round(dM[1], 4))
    print("-" * 30)
    
    print(f"Матрица dM / du(t_2):")
    print(np.round(dM[2], 4))
    print("-" * 30)