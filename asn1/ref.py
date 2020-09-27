#!/usr/local/bin/python3

import math


def printmat(A, h, k):
    print(h, k)
    for row in A:
        print(' '.join(['%5.1f' % s for s in row]))
    print('\n')

# Test 
# A = [
#     [2, 1, -1],
#     [-3, -1, 2],
#     [-2, 1, 2]
# ]
A = [
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
]

M = 3
N = 4

# Algorithm
# def ref(A, m, n):
h = 0
k = 0

while h < M and k < N:

    # Find the row index i_max
    # such that abs(A[i_max, k]) has
    # the largest magnitude
    # in that column k
    i_max = 0
    i_max_val = A[0][k]
    for i in range(h, M):
        if abs(A[i][k]) > i_max_val:
            i_max_val = abs(A[i][k])
            i_max = i
    
    printmat(A, h, k)
    if A[i_max][k] == 0:
        # No pivot in this column, go to the next column
        k += 1
    
    else:
        # Swap row with index h and index i_max
        input('swap')
        temp = A[i_max]
        A[i_max] = A[h]
        A[h] = temp
        printmat(A, h, k)

        # For each row below the pivot
        for i in range(h + 1, M):
            f = A[i][k] / A[h][k]
            A[i][k] = 0

            for j in range(k + 1, N):
                input('apply f')
                A[i][j] = A[i][j] - A[h][j] * f
                printmat(A, h, k)
        h += 1
        k += 1