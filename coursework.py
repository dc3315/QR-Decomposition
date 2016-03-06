#!/usr/bin/python

from numpy import linalg as LA
import numpy as np


# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------

# Returns the given vector, normalized.
def normalize(v):
    norm = LA.norm(v)
    if norm == 0:
        return v
    return v / norm

# Computes the projection of u onto w.
def proj(u, w):
    return (np.vdot(u, w) / (LA.norm(w)) ** 2) * w


# -----------------------------------------------------------
# QR DECOMPOSITION ALGORITHM USING GRAM SCHMIDT METHOD.
# The algorithm performs a gram schmidt decomposition on
# the matrix and concurrently calculates R.
# -----------------------------------------------------------


def qr_decomp(A):
    # Get dimension of matrix.
    n = len(A)
    # Create a copy of the matrix.
    cp = A.copy()

    # Initialize 2 matrices Q and R.
    Q = np.zeros(shape=(n, n))
    R = np.zeros(shape=(n, n))

    for i in range(n):
        u = A[:, i]
        w = u
        # Apply GS method.
        for k in range(i):
            u -= proj(w, Q[:, k])
        Q[:, i] = normalize(u)
        # Fill R at correct position.
        for j in range(i + 1):
            R[j, i] = np.vdot(Q[:, j], cp[:, i])

    # Return results.
    return Q, R


# -----------------------------------------------------------
# THE QR ITERATION ALGORITHM
# Performs iterative QR decomposition, while concurrently
# changing q to obtain the eigenvectors.
# -----------------------------------------------------------


def qr_iterator(A):
    a = []
    q_0, r_0 = qr_decomp(A)
    # Do initial computation.
    a.append(np.array(np.matrix(r_0) * np.matrix(q_0)))
    # Iterate a 100 times, changing A and q_0.
    for i in range(100):
        q, r = qr_decomp(a.pop())
        q_0 = np.matrix(q_0) * np.matrix(q)
        m = np.array(np.matrix(r) * np.matrix(q))
        a.append(m)
    # Return A and Q, modified.
    return a.pop(), q_0


# -----------------------------------------------------------
# THE MAIN METHOD.
# -----------------------------------------------------------


def main():
    # Get input from the user.
    n = int(input('Please enter the size of the matrix you would like to generate.\n'))

    # Generate random matrix of correct dimensions
    # Here N is chosen to be 100, but it can be anything.
    # The user can reset this if necessary.
    N = 100
    rand_m = np.random.uniform(-N, N, size=(n, n))

    # Make a symmetric matrix out of the generated matrix.
    A = (rand_m + rand_m.T) / 2

    # Effectuate QR iteration on A.
    q, r = qr_iterator(A.copy())

    # Open result file.
    fout = open('results.txt', 'wb')

    # Write the generated matrix.
    fout.write('The generated matrix was:\n\n'.encode('utf-8'))
    for line in np.matrix(A):
        np.savetxt(fout, line, delimiter=' ', fmt='%7.2f')

    # Write the eigenvalues.
    fout.write('\nThe resulting eigenvalues are:\n\n'.encode('utf-8'))
    eigenvalues = np.array(np.diagonal(q))
    np.savetxt(fout, eigenvalues.reshape(1, eigenvalues.shape[0]), delimiter=', ', fmt='%.2f')

    # Write the eigenvectors.
    fout.write('\nThe resulting eigenvectors are:\n\n'.encode('utf-8'))
    for line in r:
        np.savetxt(fout, line, delimiter=' ', fmt='%7.2f')

    # Finish and close the file.
    fout.close()


if __name__ == "__main__":
    main()
