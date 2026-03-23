import numpy as np
from scipy.optimize import minimize

# --- Исходные данные для Варианта 5 (I уровень сложности) ---
N = 30         # Количество моментов времени
n = 2          # Размерность вектора состояния
s = 2          # Количество неизвестных параметров
q = 4          # Количество точек спектра ( q = s(s+1)/2 + 1 )
U_MIN = 0.0    # Ограничение из варианта
U_MAX = 10.0   # Ограничение из варианта

# Истинные значения параметров
theta1 = 7.6
theta2 = 39.5

# Матрицы
F = np.array([[-theta1, 1.0], 
              [-39.5,   0.0]])
dF1 = np.array([[-1.0,  0.0], 
                [ 0.0,  0.0]])
dF2 = np.array([[ 0.0,  0.0], 
                [ 0.0,  0.0]])

Psi = np.array([[0.0], 
                [theta2]])
dPsi1 = np.array([[0.0], [0.0]])
dPsi2 = np.array([[0.0], [1.0]])
Psi_A = np.vstack((Psi, dPsi1, dPsi2))

H = np.array([[1.0, 0.0]])
Q = 0.2
R = 0.5
Gamma = np.array([[1.0], [1.0]])

P0 = np.array([[0.5, 0.0], 
               [0.0, 0.5]])

class SystemCache:
    """Класс для предвычисления независящих от U(t) матриц (Фильтр Калмана)"""
    def __init__(self):
        self.F_A_k = np.zeros((N, 6, 6))
        self.B_inv_k = np.zeros(N)
        self.M_const = np.zeros((s, s))
        self.precompute()

    def precompute(self):
        P_kk = P0.copy()
        dP_kk_1 = np.zeros((2, 2))
        
        K_k = np.zeros((2, 1))
        dK_k_1 = np.zeros((2, 1))
        
        Sigma_A = np.zeros((6, 6))
        B_k_prev = 0.0
        
        for k in range(N):
            # Построение F_A(t_k)
            K_tilde = F @ K_k
            F_minus_KH = F - K_tilde @ H
            self.F_A_k[k] = np.block([
                [F,   np.zeros((2, 2)), np.zeros((2, 2))],
                [dF1, F_minus_KH,       np.zeros((2, 2))],
                [dF2, np.zeros((2, 2)), F_minus_KH      ]
            ])
            
            # Обновление Sigma_A (форм. 22)
            K_A = np.vstack((K_k, dK_k_1, np.zeros((2, 1))))
            Sigma_A = self.F_A_k[k] @ Sigma_A @ self.F_A_k[k].T + K_A * B_k_prev @ K_A.T
            
            # Прогноз ФК
            P_k1_k = F @ P_kk @ F.T + Gamma * Q * Gamma.T
            dP_k1_k_1 = dF1 @ P_kk @ F.T + F @ dP_kk_1 @ F.T + F @ P_kk @ dF1.T
            
            # Инновация
            B_k1 = (H @ P_k1_k @ H.T)[0, 0] + R
            dB_k1_1 = (H @ dP_k1_k_1 @ H.T)[0, 0]
            
            self.B_inv_k[k] = 1.0 / B_k1
            B_k_prev = B_k1
            
            # Усиление
            K_k1 = P_k1_k @ H.T / B_k1
            dK_k1_1 = dP_k1_k_1 @ H.T / B_k1 - P_k1_k @ H.T / (B_k1**2) * dB_k1_1
            
            # Коррекция ФК
            I_KH = np.eye(2) - K_k1 @ H
            P_k1_k1 = I_KH @ P_k1_k
            dP_k1_k1_1 = -dK_k1_1 @ H @ P_k1_k + I_KH @ dP_k1_k_1
            
            # Накопление константной части ИМФ
            for i in range(s):
                for j in range(s):
                    Ci = np.zeros((2, 6)); Ci[:, (i+1)*2:(i+1)*2+2] = np.eye(2)
                    Cj = np.zeros((2, 6)); Cj[:, (j+1)*2:(j+1)*2+2] = np.eye(2)
                    
                    term1 = (H @ Ci @ Sigma_A @ Cj.T @ H.T)[0, 0] * self.B_inv_k[k]
                    
                    db_i = dB_k1_1 if i == 0 else 0.0
                    db_j = dB_k1_1 if j == 0 else 0.0
                    term2 = 0.5 * (db_i * self.B_inv_k[k] * db_j * self.B_inv_k[k])
                    
                    self.M_const[i, j] += term1 + term2
                    
            P_kk = P_k1_k1
            dP_kk_1 = dP_k1_k1_1
            K_k = K_k1
            dK_k_1 = dK_k1_1

# Инициализируем кэш системы
cache = SystemCache()

def evaluate_M_and_grad(U_vec):
    """Вычисляет M(U) и градиент dM/dU для одной траектории U"""
    M_U = cache.M_const.copy()
    dM_dU = np.zeros((N, s, s))
    
    xA = np.zeros((6, 1))
    S_hist = np.zeros((N, s, 2, 1))
    
    # Прямой проход: симуляция x_A(t_k)
    for k in range(N):
        xA = cache.F_A_k[k] @ xA + Psi_A * U_vec[k]
        S_hist[k, 0] = xA[2:4]  # d_x / d_theta1
        S_hist[k, 1] = xA[4:6]  # d_x / d_theta2
        
        for i in range(s):
            for j in range(s):
                M_U[i, j] += (H @ S_hist[k, i])[0, 0] * cache.B_inv_k[k] * (H @ S_hist[k, j])[0, 0]

    # Обратный проход для вычисления градиентов dM / dU_alpha
    for alpha in range(N):
        Phi = Psi_A
        for k in range(alpha, N):
            dC = [Phi[2:4], Phi[4:6]]
            for i in range(s):
                for j in range(s):
                    term1 = (H @ dC[i])[0,0] * cache.B_inv_k[k] * (H @ S_hist[k, j])[0,0]
                    term2 = (H @ S_hist[k, i])[0,0] * cache.B_inv_k[k] * (H @ dC[j])[0,0]
                    dM_dU[alpha, i, j] += term1 + term2
            if k < N - 1:
                Phi = cache.F_A_k[k+1] @ Phi
                
    return M_U, dM_dU

def calc_total_M(U_matrix, p):
    """Вычисляет сумму p_i * M(U_i) и все их градиенты"""
    M_total = np.zeros((s, s))
    M_list = []
    dM_list = []
    
    for i in range(q):
        M_U, dM_dU = evaluate_M_and_grad(U_matrix[i])
        M_list.append(M_U)
        dM_list.append(dM_dU)
        M_total += p[i] * M_U
        
    # Добавляем крошечную регуляризацию, чтобы избежать SingularMatrix error из-за больших чисел
    M_total += np.eye(s) * 1e-8
    return M_total, M_list, dM_list

# ===================== Критерии оптимальности =====================
def A_optimality_U(U_flat, p):
    U_matrix = U_flat.reshape((q, N))
    M_total, M_list, dM_list = calc_total_M(U_matrix, p)
    
    M_inv = np.linalg.pinv(M_total) # Используем pinv для численной стабильности
    X = np.trace(M_inv)
    
    grad = np.zeros((q, N))
    for i in range(q):
        for alpha in range(N):
            grad[i, alpha] = -p[i] * np.trace(M_inv @ dM_list[i][alpha] @ M_inv)
            
    # Масштабируем градиент, чтобы SLSQP не застрял на гигантских числах
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e10: grad = grad / (grad_norm / 1e10)
    return X, grad.flatten()

def A_optimality_p(p, U_matrix):
    M_total, M_list, _ = calc_total_M(U_matrix, p)
    
    M_inv = np.linalg.pinv(M_total)
    X = np.trace(M_inv)
    
    grad = np.zeros(q)
    for i in range(q):
        grad[i] = -np.trace(M_inv @ M_list[i] @ M_inv)
        
    return X, grad

def D_optimality_U(U_flat, p):
    U_matrix = U_flat.reshape((q, N))
    M_total, M_list, dM_list = calc_total_M(U_matrix, p)
    
    sign, logdet = np.linalg.slogdet(M_total) # slogdet спасает от inf
    if sign <= 0: return 1e6, np.zeros(q*N)
        
    M_inv = np.linalg.pinv(M_total)
    X = -logdet
    
    grad = np.zeros((q, N))
    for i in range(q):
        for alpha in range(N):
            grad[i, alpha] = -p[i] * np.trace(M_inv @ dM_list[i][alpha])
            
    # Масштабируем градиент
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e10: grad = grad / (grad_norm / 1e10)
    return X, grad.flatten()

def D_optimality_p(p, U_matrix):
    M_total, M_list, _ = calc_total_M(U_matrix, p)
    
    sign, logdet = np.linalg.slogdet(M_total)
    if sign <= 0: return 1e6, np.zeros(q)
        
    M_inv = np.linalg.pinv(M_total)
    X = -logdet
    
    grad = np.zeros(q)
    for i in range(q):
        grad[i] = -np.trace(M_inv @ M_list[i])
        
    return X, grad

# ================= Прямая процедура (SQP Алгоритм) =================
def run_direct_procedure(opt_type="A"):
    # Шаг 1. Задаем начальный невырожденный план (равномерно в [0, 10])
    U_matrix = np.array([np.ones(N)*1.0, np.ones(N)*4.0, np.ones(N)*7.0, np.ones(N)*10.0])
    p = np.array([0.25, 0.25, 0.25, 0.25])
    
    if opt_type == "A":
        X_init, _ = A_optimality_p(p, U_matrix)
    else:
        X_init, _ = D_optimality_p(p, U_matrix)
        
    bounds_U = [(U_MIN, U_MAX)] * (q * N)
    bounds_p = [(0.0, 1.0)] * q
    cons_p = {'type': 'eq', 'fun': lambda p: np.sum(p) - 1.0}
    
    MAX_ITER = 3 # 3 итерации пересчета SQP
    
    for iteration in range(MAX_ITER):
        # Шаг 2. Оптимизация U (сигналов) при фиксированных весах p
        opt_U_func = A_optimality_U if opt_type == "A" else D_optimality_U
        res_U = minimize(opt_U_func, U_matrix.flatten(), args=(p,), 
                         method='SLSQP', jac=True, bounds=bounds_U, 
                         options={'ftol': 1e-4, 'disp': False})
        U_matrix = res_U.x.reshape((q, N))
        
        # Шаг 3. Оптимизация p (весов) при фиксированных сигналах U
        opt_p_func = A_optimality_p if opt_type == "A" else D_optimality_p
        res_p = minimize(opt_p_func, p, args=(U_matrix,), 
                         method='SLSQP', jac=True, bounds=bounds_p, constraints=cons_p, 
                         options={'ftol': 1e-4, 'disp': False})
        p = res_p.x
        
    # Итоговый критерий
    if opt_type == "A":
        X_final, _ = A_optimality_p(p, U_matrix)
    else:
        X_final, _ = D_optimality_p(p, U_matrix)
        
    return X_init, X_final

# ======================== Вывод результатов ========================
if __name__ == "__main__":
    print("Выполнение прямой процедуры синтеза (Вариант 5)... Пожалуйста, подождите.\n")
    print("Примечание: из-за неустойчивости матрицы F значения критериев (след/детерминант)\n"
          "могут иметь экспоненциальный масштаб.\n")
    
    # Синтез A-оптимального плана
    A_init, A_final = run_direct_procedure(opt_type="A")
    
    # Синтез D-оптимального плана
    D_init, D_final = run_direct_procedure(opt_type="D")
    
    print("-" * 75)
    print(f"| {'Исходный план':<20} | {'Значение критерия':<20} | {'Значения критерия':<23} |")
    print(f"| {'(тип)':<20} | {'(до синтеза)':<20} | {'(после синтеза)':<23} |")
    print("-" * 75)
    print(f"| {'A-оптимальности':<20} | {A_init:<20.4e} | {A_final:<23.4e} |")
    print(f"| {'D-оптимальности':<20} | {D_init:<20.4e} | {D_final:<23.4e} |")
    print("-" * 75)