from tkinter.tix import Tree
import numpy as np


def main():

    omega = 0.5
    eps = 0.001
    
    A = np.array(
        [
            [8, 1, 2],
            [6, 9, 5],
            [1, 5, 9],
        ]
    )

    f = np.array(
        [
            2,
            5,
            6
        ]
    )
    
    x = relaxation(A, f, omega, eps)

    print(x)



def relaxation(A, f, omega, eps):
    
    n = A.shape[0]    # Размер вектора решения
    x = np.zeros(n)     # Начальное приближеение
    
    while True:

        x_prev = x.copy()

        for j in range(n):

            local_sum = 0

            for k in range(n):
                local_sum += A[j, k] / A[j, j] * x[k]
                
            x[j] = x[j] - omega * local_sum + omega * f[j] / A[j, j]

        # Условие остановки
        if np.linalg.norm(x_prev - x) < eps:
            break
    
    return x


if __name__ == "__main__":
    main()