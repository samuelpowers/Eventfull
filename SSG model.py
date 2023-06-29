# This code is associated with the model presented in the section "Spin systems in rotated frames"

import scipy.special
import math
from math import factorial as mf
import numpy as np
import matplotlib.pyplot as plt

# Define conditioning variables (U)
# n should not exceed 150. Computational time increases as n and j increase, and |m_a1| decreases.
n = 20
j = 1
m_a1 = 0

# theta will be summed over

# tuning parameter
x = 0
# 0.1377 for good fit

# Coefficient array calculated by Mathematica
# Each row of the coefficient matrix is associated with a count (AA,AB,BA,BB,CC,CD,DC,DD)
# Each column of the matrix is associated with a quantum number (n,j,ma1,mb2,la1,lb2,B_map,DD)
# We use DD rather than nu_0 in this program because its range is simpler to define (see counter function)
# nu_0 is then calculated within the cardinality functions for Alice's and Bob's elementary state spaces
coefficient_array = np.asarray([[1/2, 0, -(1/2), -(1/2), 1/2, 1/2, -(1/2), -1],
                                [0, -1, 1/2, 1/2, 1/2, -(1/2), 1/2, 1],
                                [0, -1, 1/2, 1/2, -(1/2), 1/2, 1/2, 1],
                                [1/2, 0, -(1/2), -(1/2), -(1/2), -(1/2), -(1/2), -1],
                                [0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 1, 0, -1, 0, 0, 0, -1],
                                [0, 1, -1, 0, 0, 0, 0, -1],
                                [0, 0, 0, 0, 0, 0, 0, 1]])


# Calculates the cardinality of Alice's and Bob's elementary state spaces (\varepsilon).
# Also calculates nu_0
def card_alice_vec(count_array_):
    nu_0 = (count_array_[:, 0] + count_array_[:, 3] + count_array_[:, 5] + count_array_[:, 6])/4

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Alice's event
    A_a1 = count_array_[:, 0] + count_array_[:, 1]
    B_a1 = count_array_[:, 2] + count_array_[:, 3]
    C_a1 = count_array_[:, 4] + count_array_[:, 5]
    D_a1 = count_array_[:, 6] + count_array_[:, 7]

    # Cardinality of \varepsilon^a. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_a1, exact=x) / (
                scipy.special.factorial(count_array_[:, 0], exact=x) * scipy.special.factorial(count_array_[:, 4], exact=x))
    x_b = scipy.special.factorial(B_a1, exact=x) / (
                scipy.special.factorial(count_array_[:, 1], exact=x) * scipy.special.factorial(count_array_[:, 5], exact=x))
    x_c = scipy.special.factorial(C_a1, exact=x) / (
                scipy.special.factorial(count_array_[:, 2], exact=x) * scipy.special.factorial(count_array_[:, 6], exact=x))
    x_d = scipy.special.factorial(D_a1, exact=x) / (
                scipy.special.factorial(count_array_[:, 3], exact=x) * scipy.special.factorial(count_array_[:, 7], exact=x))

    cardinality_alice = x_a * x_b * x_c * x_d

    return cardinality_alice, nu_0


def card_bob_vec(count_array_):
    nu_0 = (count_array_[:, 0] + count_array_[:, 3] + count_array_[:, 5] + count_array_[:, 6]) / 4

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Bob's event
    A_b2 = count_array_[:, 0] + count_array_[:, 2]
    B_b2 = count_array_[:, 1] + count_array_[:, 3]
    C_b2 = count_array_[:, 4] + count_array_[:, 6]
    D_b2 = count_array_[:, 5] + count_array_[:, 7]

    # Cardinality of  \varepsilon^b. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_b2, exact=x) / (
                scipy.special.factorial(count_array_[:, 0], exact=x) * scipy.special.factorial(count_array_[:, 4], exact=x))
    x_b = scipy.special.factorial(B_b2, exact=x) / (
                scipy.special.factorial(count_array_[:, 1], exact=x) * scipy.special.factorial(count_array_[:, 5], exact=x))
    x_c = scipy.special.factorial(C_b2, exact=x) / (
                scipy.special.factorial(count_array_[:, 2], exact=x) * scipy.special.factorial(count_array_[:, 6], exact=x))
    x_d = scipy.special.factorial(D_b2, exact=x) / (
                scipy.special.factorial(count_array_[:, 3], exact=x) * scipy.special.factorial(count_array_[:, 7], exact=x))

    cardinality_bob = x_a * x_b * x_c * x_d

    return cardinality_bob, nu_0


# Calculates the number of configurations of the symbols A and B in Alice's event
def config_alice_ab(count_array_):

    A_a1 = count_array_[0] + count_array_[1]
    B_a1 = count_array_[2] + count_array_[3]

    return mf(A_a1 + B_a1)/(mf(A_a1)*mf(B_a1))


# Calculates the number of configurations of the symbols A and B in Bob's event
def config_bob_ab(count_array_):

    A_b2 = count_array_[0] + count_array_[2]
    B_b2 = count_array_[1] + count_array_[3]

    return mf(A_b2 + B_b2) / (mf(A_b2) * mf(B_b2))


# Calculates the cardinality of Alice's and Bob's joint state space for a given set of local quantum numbers
def counter(n_, j_, m_a1_, m_b2_, l_a1_, l_b2_, B_map_, coefficient_array_):
    # Many of the unique combinations of quantum numbers being summed over in this function will not be valid.
    # This can be avoided by carefully considering the allowed ranges of the quantum numbers being summed over.
    # Defining these ranges would significantly improve the speed of this calculation, but slows down development.
    # In this code, we initially include the invalid combinations and remove them towards the end.

    # Initialize the count for Alice's and Bob's local state spaces
    alice_local_state_space = 0
    bob_local_state_space = 0

    # Set the default for Alice's and Bob's event configurations
    alice_config = 1
    bob_config = 1

    # Define DD range
    dd_range_ = np.arange(0, min(j_ - m_a1_, j_ - m_b2_) + 1, 1)

    # Define array to store quantum numbers (to be used with the coefficient array to calculate counts)
    q_number_array = np.ndarray(shape=(8, 8))

    # Define the count array, which stores the counts associated with every complete set of quantum numbers
    count_array = []

    # Sum over the non-local nuisance variable.
    for dd in range(len(dd_range_)):
        # Fill the quantum number array with the chosen values for the complete set of quantum numbers
        q_number_array[:] = n_, j_, m_a1_, m_b2_, l_a1_, l_b2_, B_map_, dd_range_[dd]

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


# Calculates the probability of all combinations of random variable m_b2, given B_map
def prob(n_, j_, m_a1_, B_map_, coefficient_array_):
    # get the allowed range of m_b2
    m_b2_range = np.arange(-j_, j_ + 1)

    # define arrays to store data
    label_array_ = []
    upsilon_array_ = np.zeros(shape=(len(m_b2_range)))
    index_ = 0

    # iterate over random variable
    for b in range(len(m_b2_range)):
        # store the labels
        label_array_.append(str(m_a1_) + "  " + str(m_b2_range[b]))

        # Define l_a1 and l_b2 ranges
        l_a1_range_ = np.arange(-(n_ / 2 - j_), (n_ / 2 - j_) + 1, 1)
        l_b2_range_ = np.arange(-(n_ / 2 - j_), (n_ / 2 - j_) + 1, 1)

        # sum over local nuisance variables
        for c in range(len(l_a1_range_)):
            for d in range(len(l_b2_range_)):
                # Calculate the contribution to Upsilon for this set of local quantum numbers
                upsilon_array_[index_] += counter(n_, j_, m_a1_, m_b2_range[b], l_a1_range_[c], l_b2_range_[d], B_map_, coefficient_array_)

        # move on to the next m_b2
        index_ += 1

    # normalize
    probability = upsilon_array_ / np.sum(upsilon_array_)
    return probability, label_array_


# Calculates the probability distribution as a function of B_map for both the new model and QM
def prob_ssg(n_, j_, m_a1_, coefficient_array_, x_):
    # Define arrays that will store data
    probability_data = []
    wigner_data = []
    theta_data = []
    diff_data = []

    # Initialize b_map at 0
    B_map_ = 0

    # Get labels
    labels = prob(n_, j_, m_a1_, B_map_, coefficient_array_)[1]
    # Sum over all values of b_map, from 0 to n
    while B_map_ <= n_:
        # Print the current value of b_map/n so the user can track progress
        print("B_map_= ", B_map_, "/", n)

        theta = math.pi * B_map_ / n_ - x_*math.sin(2*math.pi * B_map_ / n_)

        temp_new = prob(n_, j_, m_a1_, B_map_, coefficient_array_)[0]
        temp_wig = wigner_formula(j_, m_a1_, theta)

        probability_data.append(temp_new)
        wigner_data.append(temp_wig)
        diff_data.append(abs(temp_new - temp_wig))

        theta_data.append(theta)

        B_map_ += 1

    return theta_data, probability_data, wigner_data, diff_data, labels

# Calculates the probability distribution as a function of theta_ab in QM
def wigner_formula(j_, m_, theta_):
    mp_range = np.arange(-j_, j_ + 1)
    # This expression is given in Sakurai (Wigner formula)
    # The conditions are in place to ensure none of the combinatorial arguments become negative.
    wigner = np.zeros(shape=(len(mp_range)))

    for i in range(len(mp_range)):
        mp_ = mp_range[i]
        q_min = max(0, m_ - mp_)
        q_max = min(j_ - mp_, j_ + m_)
        q = q_min

        while q <= q_max:
            if j_ + m_ - q >= 0:
                if q >= 0:
                    if mp_ - m_ + q >= 0:
                        if j_ - mp_ - q >= 0:
                            term_1 = mf(j_ + m_) * mf(j_ - m_) * mf(j_ + mp_) * mf(j_ - mp_)
                            term_2 = mf(j_ + m_ - q) * mf(q) * mf(mp_ - m_ + q) * mf(j_ - mp_ - q)
                            term_3 = (math.cos(theta_ / 2) ** (2 * j_ + m_ - mp_ - 2 * q)) * (
                                    math.sin(theta_ / 2) ** (mp_ - m_ + 2 * q))
                            wigner[i] += ((-1) ** (mp_ - m_ + q)) * (math.sqrt(term_1) / term_2) * term_3
            q += 1
    # We then square the result to obtain a probability (Born rule)
    return wigner ** 2


# Execute the calculation and store the data for plotting
data = prob_ssg(n, j, m_a1, coefficient_array, x)

# Generate plots
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex='all')
fig.subplots_adjust(hspace=.00)
axs[0].plot(data[0], data[2], c="blue", label="QM")
axs[0].plot(data[0], data[1], c="red", label='New Model')

plot_title = "n=" + str(n) + r", j=" + str(j) + r", $m_{a1}$=" + str(m_a1)


np.save(plot_title + '.npy', data)

axs[0].set(ylabel='Probability', title=plot_title)
axs[0].set_ylim(-0.05, 1.05)
axs[0].legend()

axs[1].plot(data[0], data[3])
axs[1].set(xlabel=r'$\theta$ (radians)', ylabel=r'|$\Delta$|')
axs[1].set_xlim(0, math.pi)
axs[1].set_ylim(-0.005, np.max(data[3])+.005)


plt.show()


