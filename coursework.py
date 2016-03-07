#!/usr/bin/python

# -----------------------------------------------------------
# IMPORTS.
# Foreword: I imported numpy and linalg libraries,
# only to use functions that I do not deem to useful
# to rewrite. Its not worth reinventing the wheel for
# certain functions such as vdot (dot product of two vectors)
# and norm. These can be easily be rewritten if necessary
# but I did not do so since I did not want to waste time and
# instead focused on the core part of the exercise.
# -----------------------------------------------------------


from numpy import linalg as LA
import numpy as np


# -----------------------------------------------------------
# PRECISION THRESHOLD CONSTANT EPSILON
# The threshold is set to 0.0001 but can be set to a
# smaller value if desired.
# -----------------------------------------------------------


EPSILON = 1e-4


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
# changing q to obtain the eigenvectors. The function
# uses a helper which determines whether the computation
# is precise enough.
# -----------------------------------------------------------


def accurate_computation(r_0, r_1):
    subtraction = np.matrix(r_1) - np.matrix(r_0)
    return LA.norm(subtraction) < EPSILON


def qr_iterator(A):
    q_0, r_0 = qr_decomp(A)
    # Do initial computation.
    a = np.array(np.matrix(r_0) * np.matrix(q_0))
    # Iterate until the computation is accurate enough.
    while True:
        q, r = qr_decomp(a)
        if accurate_computation(r_0, r):
            return a, q_0
        else:
            q_0 = np.matrix(q_0) * np.matrix(q)
            m = np.array(np.matrix(r) * np.matrix(q))
            r_0 = r
            a = m


# -----------------------------------------------------------
# THE MAIN METHOD.
# -----------------------------------------------------------


def main():

    # Get input from the user.
    n = int(input('Please enter the size of the matrix you would like to generate.\n'))

    # Generate random matrix of correct dimensions
    # Here N is chosen to be 100, but it can be any value
    # the user sees fit.
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
        np.savetxt(fout, line, delimiter=' ', fmt='%8.3f')

    # Write the eigenvalues.
    fout.write('\nThe resulting eigenvalues are:\n\n'.encode('utf-8'))
    eigenvalues = np.array(np.diagonal(q))
    np.savetxt(fout, eigenvalues.reshape(1, eigenvalues.shape[0]), delimiter=', ', fmt='%.3f')

    # Write the eigenvectors.
    fout.write('\nThe resulting eigenvectors are:\n\n'.encode('utf-8'))
    for line in r:
        np.savetxt(fout, line, delimiter=' ', fmt='%8.3f')

    # Finish and close the file.
    fout.close()


if __name__ == "__main__":
    main()
