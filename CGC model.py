import scipy.special
import math
from math import factorial as mf
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum.cg import CG
from sympy import S

# Define conditioning variables (U)
# n should not exceed 150. Computational time increases as n and j increase, and |m_a1| decreases.
n = 50

j_1a = 1
j_b2 = 1

j_12 = 0

# Define m_12 to enforce selection rule
m_12 = 0


# Coefficient array calculated by Mathematica
# Each row of the coefficient matrix is associated with a count (AA,AD,BB,BC,CB,CC,DA,DD)
# Each column of the matrix is associated with a quantum number (n,j1a,m1a,jb2,mb2,j12,l12,DD)
# We use DD rather than nu_0 in this program because its range is simpler to define (see counter function)
# nu_4 is then calculated within the cardinality functions for Alice's and Bob's elementary state spaces
coefficient_array = np.asarray([[1/2, -1, 0, -1, 0, 0, 1, 1],
                                [0, 0, 0, 1, -1, 0, 0, -1],
                                [1/2, 0, 0,0, 0, -1, -1, -1],
                                [0, -1, 0, 0, 1, 1, 0, 1],
                                [0, 0, -1, -1, 0, 1, 0, 1],
                                [0, 1, 0, 1, 0, -1, 0, -1],
                                [0, 1, 1, 0, 0, 0, 0, -1],
                                [0, 0,0, 0, 0, 0, 0, 1]])


# Calculates the cardinality of Alice's and Bob's elementary state spaces (\varepsilon).
# Also calculates nu_4
def card_alice_vec(count_array_):
    nu_4 = (count_array_[:, 0] + count_array_[:, 3] + count_array_[:, 4] + count_array_[:, 7])/4

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Alice's event
    A_a1 = count_array_[:, 0] + count_array_[:, 1]
    B_a1 = count_array_[:, 2] + count_array_[:, 3]
    C_a1 = count_array_[:, 4] + count_array_[:, 5]
    D_a1 = count_array_[:, 6] + count_array_[:, 7]

    # Cardinality of \varepsilon^a. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_a1, exact=x) / (
            scipy.special.factorial(count_array_[:, 0], exact=x) * scipy.special.factorial(count_array_[:, 1], exact=x))
    x_b = scipy.special.factorial(B_a1, exact=x) / (
            scipy.special.factorial(count_array_[:, 2], exact=x) * scipy.special.factorial(count_array_[:, 3], exact=x))
    x_c = scipy.special.factorial(C_a1, exact=x) / (
            scipy.special.factorial(count_array_[:, 4], exact=x) * scipy.special.factorial(count_array_[:, 5], exact=x))
    x_d = scipy.special.factorial(D_a1, exact=x) / (
            scipy.special.factorial(count_array_[:, 6], exact=x) * scipy.special.factorial(count_array_[:, 7], exact=x))

    cardinality_alice = x_a * x_b * x_c * x_d

    return cardinality_alice, nu_4


def card_bob_vec(count_array_):
    nu_4 = (count_array_[:, 0] + count_array_[:, 3] + count_array_[:, 4] + count_array_[:, 7]) / 4

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Bob's event
    A_b2 = count_array_[:, 0] + count_array_[:, 6]
    B_b2 = count_array_[:, 2] + count_array_[:, 4]
    C_b2 = count_array_[:, 3] + count_array_[:, 5]
    D_b2 = count_array_[:, 1] + count_array_[:, 7]

    # Cardinality of  \varepsilon^b. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_b2, exact=x) / (
            scipy.special.factorial(count_array_[:, 0], exact=x) * scipy.special.factorial(count_array_[:, 6], exact=x))
    x_b = scipy.special.factorial(B_b2, exact=x) / (
            scipy.special.factorial(count_array_[:, 2], exact=x) * scipy.special.factorial(count_array_[:, 4], exact=x))
    x_c = scipy.special.factorial(C_b2, exact=x) / (
            scipy.special.factorial(count_array_[:, 3], exact=x) * scipy.special.factorial(count_array_[:, 5], exact=x))
    x_d = scipy.special.factorial(D_b2, exact=x) / (
            scipy.special.factorial(count_array_[:, 1], exact=x) * scipy.special.factorial(count_array_[:, 7], exact=x))

    cardinality_bob = x_a * x_b * x_c * x_d

    return cardinality_bob, nu_4


def config_alice_ab(count_array_):

    A_a1 = count_array_[0] + count_array_[1]
    B_a1 = count_array_[2] + count_array_[3]

    return mf(A_a1 +  B_a1)/(mf(A_a1)*mf(B_a1))


def config_bob_ab(count_array_):

    A_b2 = count_array_[0] + count_array_[6]
    B_b2 = count_array_[2] + count_array_[4]

    return mf(A_b2 + B_b2) / (mf(A_b2) * mf(B_b2))


# Calculates the cardinality of Alice's and Bob's joint state space for a given set of local quantum numbers
def counter(n_, j_1a_, m_1a_, j_b2_, m_b2_, j_12_, l_12_, coefficient_array_):
    # Many of the unique combinations of quantum numbers being summed over in this function will not be valid.
    # This can be avoided by carefully considering the allowed ranges of the quantum numbers being summed over.
    # Defining these ranges would significantly improve the speed of this calculation, but slows down development.
    # In this code, we initially include the invalid combinations and remove them towards the end.

    # Initialize the count
    alice_local_state_space = 0
    bob_local_state_space = 0

    # Set the default for Alice's and Bob's event configurations
    alice_config = 1
    bob_config = 1

    # Define DD range
    dd_range_ = np.arange(0, min(j_1a_ + m_1a_, j_b2_ - m_b2_) + 1, 1)

    # Define array to store quantum numbers (to be used with the coefficient array to calculate counts)
    q_number_array = np.ndarray(shape=(8, 8))

    # Define the count array, which stores the counts associated with every complete set of quantum numbers
    count_array = []

    # Sum over the non-local nuisance variable.
    for dd in range(len(dd_range_)):

        q_number_array[:] = n_, j_1a_, m_1a_, j_b2_, m_b2_, j_12_, l_12_, dd_range_[dd]

        # Calculate the associated counts
        count_array.append(np.sum(q_number_array * coefficient_array_, 1))

    # Make sure the count array is valid. This is where we remove invalid combinations of quantum numbers.
    count_array = np.asarray(count_array)
    count_array_groomed = []

    # Identify situations in which some error has prevented a single set of counts from being defined.
    if len(count_array) > 0:

        # Identify the invalid entries in the count array.
        # Inner condition 1 requires that each count is greater than 0
        # Inner condition 2 requires that each count is an integer
        # If both conditions are met for all 8 counts, the outer condition is met.
        mask = np.where(np.sum(np.where((count_array >= 0) & (count_array % 1 == 0), 1, 0), 1) == 8, 1, 0)

        # Fill a new array with only the sets of counts that were determined to be valid.
        for t in range(len(mask)):
            if mask[t] == 1:
                count_array_groomed.append(count_array[t])

        # Identify situations in which there are no valid sets of counts
        if len(count_array_groomed) > 0:
            count_array_groomed = np.asarray(count_array_groomed)

            # Use the valid sets of counts to calculate the cardinality of Alice's and Bob's local state spaces
            phi_alice_ = card_alice_vec(count_array_groomed)
            phi_bob_ = card_bob_vec(count_array_groomed)

            # Now that we know we've found a valid set of quantum numbers, get the number of configurations of the
            # symbols A and B in Alice's and Bob's events.
            alice_config = config_alice_ab(count_array_groomed[0])
            bob_config = config_bob_ab(count_array_groomed[0])

            # Calculate the cardinalities of Alice's and Bob's local state spaces
            for v in range(len(count_array_groomed)):
                alice_local_state_space += phi_alice_[0][v] * (-1) ** (abs(phi_alice_[1][0] - phi_alice_[1][v]))
                bob_local_state_space += phi_bob_[0][v] * (-1) ** (abs(phi_bob_[1][0] - phi_bob_[1][v]))

    # return the total number of ordered pairs of states in Alice's and Bob's product space (after interference).
    # scale by the number of configurations of the symbols A and B in Alice's and Bob's event
    return alice_local_state_space * bob_local_state_space * alice_config * bob_config


# Calculates the probability of all combinations of random variable m_b2 and m_a1
def prob_cgc(n_, j_1a_, j_b2_, j_12_, m_12_, coefficient_array_):

    # define ranges for random variables
    m_1a_range_ = np.arange(-j_1a_, j_1a_ + 1, 1)
    m_b2_range_ = np.arange(-j_b2_, j_b2_ + 1, 1)

    # define arrays to store data
    label_array_ = []
    upsilon_array_ = np.zeros(shape=(len(m_1a_range_)*len(m_b2_range_)))
    cg_array = np.zeros(shape=(len(m_1a_range_)*len(m_b2_range_)))
    index_ = 0

    # Define the range for l_12
    l_12_range_ = np.arange(-(n_ / 2 - j_12_), (n_ / 2 - j_12_) + 1, 1)

    # iterate over random variables
    for q in range(len(m_1a_range_)):
        for r in range(len(m_b2_range_)):
            # store the labels
            label_array_.append(str(m_1a_range_[q]) + "  " + str(m_b2_range_[r]))

            # Apply m_12 constraint
            if m_1a_range_[q] + m_b2_range_[r] == m_12_:

                # Sum over the local nuisance variable
                for a in range(len(l_12_range_)):

                    # if l_12 is a nuisance variable, determine the number of local state space configurations
                    upsilon_array_[index_] += counter(n_, j_1a_, m_1a_range_[q], j_b2_, m_b2_range_[r], j_12_, l_12_range_[a], coefficient_array_)
                    cg = CG(S(2 * j_1a_) / 2, S(2 * m_1a_range_[q]) / 2, S(2 * j_b2_) / 2, S(2 * m_b2_range_[r]) / 2, S(2 * j_12_) / 2, S(2 * m_12_) / 2)
                    cg_array[index_] = float((cg.doit())**2)


            # move on to the next m_1a and m_b2
            index_ += 1

    # normalize
    probability = upsilon_array_ / np.sum(upsilon_array_)
    return probability, cg_array, label_array_


# Execute the calculation and store the data for plotting
data = prob_cgc(n, j_1a, j_b2, j_12, m_12, coefficient_array)

for k in range(len(data[0])):
    print(data[2][k], ": ", data[0][k], data[1][k])


def n_function(n_max_, coefficient_array_):
    n_ = 3
    data_diff = []

    j_1a_ = 1
    j_b2_ = 1

    j_12_ = 1
    m_12_ = 0

    while n_ <= n_max_:
        print(n_)
        data_diff.append(abs(prob_cgc(n_, j_1a_, j_b2_, j_12_, m_12_, coefficient_array_)[1]-prob_cgc(n_, j_1a_, j_b2_, j_12_, m_12_, coefficient_array_)[0]))
        n_ += 1

    plot_title = "n_maz=" + str(n_max_) + " CGC Diff Data"

    np.save(plot_title + '.npy', data_diff)

    return data_diff


# diff_data = n_function(100, coefficient_array)
# print(diff_data)
# plt.plot(diff_data)

# plt.show()

