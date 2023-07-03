# This code is associated with the model presented in the section "Bell Test"

import scipy.special
import math
from math import factorial as mf
import numpy as np
import matplotlib.pyplot as plt
import time


# Define conditioning variables (U)
# n should not exceed 100. Computational time increases as n and j increase, and |m| decreases.
n = 50

j_1a = 1/2
j_b2 = 1/2

j_gamma = 0
# constraint
m_gamma = 0

alpha_map = 4

# tuning parameter
x = 0
# 0.1377 for good fit

# Coefficient array calculated by Mathematica
# Each row of the coefficient matrix is associated with a count (AA,AB,AC,AD,BA,BB,BC,BD,CA,CB,CC,CD,DA,DB,DC,DD)
# Each column of the matrix is associated with (n,j1a,m1a,l1a,jb2,mb2,lb2,alpha,beta,gamma,AC,AD,CC,CA,DD,DC)
# We use AC,AD,CC,CA,DD,DC rather than nu0, nu1, nu4, nu5, k1, w2 in this program
# We do this because their ranges are simpler to define (see path_counter function)
# The quantum numbers nu0, nu1, nu4, nu5, k1, w2 are then calculated within the cardinality functions for Alice and Bob
coefficient_array = np.asarray([[1/2, 0, -(1/2), 1/2, 0, 1/2, 1/2, 0, -(1/2), 0, -1, 0, -1, -1, 0, 0],
                                [0, -1, 1/2, 1/2, 0, -(1/2), -(1/2), 0, 1/2, 0, 0, -1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                [0, 0, -(1/2), -(1/2), -1, 1/2, 1/2, 1/2, 0, 0, 0, 1, 0, -1, 1, 0],
                                [1/2, 0, 1/2, -(1/2), 0, -(1/2), -(1/2), -(1/2), 0, -(1/2), 1, 0, 0, 1, -1, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, -1],
                                [0, -1, 0, 0, 0, -1, 0, 0, 0, 1/2, 0, -1, 1, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, -1, 0, -1, 0, 0, 0, 0, 1/2, 0, 0, 0, -1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 1, 0, 0, 1, 0, 0, 0, 0, -(1/2), 0, 0, -1, 0, -1, -1],
                                [0, 0, 1, 0, 0, -1, 0, -(1/2), 1/2, 0, 1, -1, 1, 1, -1, 0],
                                [0, 1, 0, 0, 0, 1, 0, 1/2, -(1/2), 0, -1, 1, -1, -1, 0, -1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])


# Calculates the cardinality of the ontic states in Alice's and Bob's sub-ensembles.
# Also calculates non-local quantum numbers
def card_alice_vec(count_array_):
    # The non-local quantum numbers
    k_1_ = (count_array_[:, 2] + count_array_[:, 5] + count_array_[:, 11] + count_array_[:, 12]) / 4
    w_2_ = (count_array_[:, 2] + count_array_[:, 4] + count_array_[:, 9] + count_array_[:, 15]) / 4
    nu_0_ = (count_array_[:, 0] + count_array_[:, 5] + count_array_[:, 11] + count_array_[:, 14]) / 4
    nu_1_ = (count_array_[:, 3] + count_array_[:, 6] + count_array_[:, 8] + count_array_[:, 13]) / 4
    nu_4_ = (count_array_[:, 0] + count_array_[:, 6] + count_array_[:, 9] + count_array_[:, 15]) / 4
    nu_5_ = (count_array_[:, 2] + count_array_[:, 4] + count_array_[:, 11] + count_array_[:, 13]) / 4

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Alice's event
    A_a1 = count_array_[:, 0] + count_array_[:, 1] + count_array_[:, 2] + count_array_[:, 3]
    B_a1 = count_array_[:, 4] + count_array_[:, 5] + count_array_[:, 6] + count_array_[:, 7]
    C_a1 = count_array_[:, 8] + count_array_[:, 9] + count_array_[:, 10] + count_array_[:, 11]
    D_a1 = count_array_[:, 12] + count_array_[:, 13] + count_array_[:, 14] + count_array_[:, 15]

    # Cardinality of \varepsilon^a. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_a1, exact=x) / (scipy.special.factorial(count_array_[:, 0], exact=x) * scipy.special.factorial(count_array_[:, 1], exact=x)*scipy.special.factorial(count_array_[:, 2], exact=x) * scipy.special.factorial(count_array_[:, 3], exact=x))
    x_b = scipy.special.factorial(B_a1, exact=x) / (scipy.special.factorial(count_array_[:, 4], exact=x) * scipy.special.factorial(count_array_[:, 5], exact=x)*scipy.special.factorial(count_array_[:, 6], exact=x) * scipy.special.factorial(count_array_[:, 7], exact=x))
    x_c = scipy.special.factorial(C_a1, exact=x) / (scipy.special.factorial(count_array_[:, 8], exact=x) * scipy.special.factorial(count_array_[:, 9], exact=x)*scipy.special.factorial(count_array_[:, 10], exact=x) * scipy.special.factorial(count_array_[:, 11], exact=x))
    x_d = scipy.special.factorial(D_a1, exact=x) / (scipy.special.factorial(count_array_[:, 12], exact=x) * scipy.special.factorial(count_array_[:, 13], exact=x)*scipy.special.factorial(count_array_[:, 14], exact=x) * scipy.special.factorial(count_array_[:, 15], exact=x))

    cardinality_alice = x_a * x_b * x_c * x_d

    # return non-local quantum numbers individually to facilitate m_\gamma constraint

    return cardinality_alice, k_1_, w_2_, nu_0_, nu_1_, nu_4_, nu_5_, k_1_ + w_2_ + nu_0_ + nu_1_ + nu_4_ + nu_5_


def card_bob_vec(count_array_):
    # The non-local quantum numbers
    k_1_ = (count_array_[:, 2] + count_array_[:, 5] + count_array_[:, 11] + count_array_[:, 12]) / 4
    w_2_ = (count_array_[:, 2] + count_array_[:, 4] + count_array_[:, 9] + count_array_[:, 15]) / 4
    nu_0_ = (count_array_[:, 0] + count_array_[:, 5] + count_array_[:, 11] + count_array_[:, 14]) / 4
    nu_1_ = (count_array_[:, 3] + count_array_[:, 6] + count_array_[:, 8] + count_array_[:, 13]) / 4
    nu_4_ = (count_array_[:, 0] + count_array_[:, 6] + count_array_[:, 9] + count_array_[:, 15]) / 4
    nu_5_ = (count_array_[:, 2] + count_array_[:, 4] + count_array_[:, 11] + count_array_[:, 13]) / 4

    # x = True bypasses numerical approximations used in scipy.special.factorial. This increases computational time.
    x = False

    # The base-4 counts associated with Bob's event
    A_b2 = count_array_[:, 0] + count_array_[:, 4] + count_array_[:, 8] + count_array_[:, 12]
    B_b2 = count_array_[:, 1] + count_array_[:, 5] + count_array_[:, 9] + count_array_[:, 13]
    C_b2 = count_array_[:, 2] + count_array_[:, 6] + count_array_[:, 10] + count_array_[:, 14]
    D_b2 = count_array_[:, 3] + count_array_[:, 7] + count_array_[:, 11] + count_array_[:, 15]

    # Cardinality of \varepsilon^b. These are broken into separate terms to avoid overflow as much as possible.
    x_a = scipy.special.factorial(A_b2 + B_b2, exact=x) / (scipy.special.factorial(count_array_[:, 0], exact=x)*scipy.special.factorial(count_array_[:, 4], exact=x)*scipy.special.factorial(count_array_[:, 8], exact=x)*scipy.special.factorial(count_array_[:, 12], exact=x))
    x_b = scipy.special.factorial(0, exact=x) / (scipy.special.factorial(count_array_[:, 1], exact=x)*scipy.special.factorial(count_array_[:, 5], exact=x)*scipy.special.factorial(count_array_[:, 9], exact=x)*scipy.special.factorial(count_array_[:, 13], exact=x))
    x_c = scipy.special.factorial(C_b2, exact=x) / (scipy.special.factorial(count_array_[:, 2], exact=x)*scipy.special.factorial(count_array_[:, 6], exact=x)*scipy.special.factorial(count_array_[:, 10], exact=x)*scipy.special.factorial(count_array_[:, 14], exact=x))
    x_d = scipy.special.factorial(D_b2, exact=x) / (scipy.special.factorial(count_array_[:, 3], exact=x)*scipy.special.factorial(count_array_[:, 7], exact=x)*scipy.special.factorial(count_array_[:, 11], exact=x)*scipy.special.factorial(count_array_[:, 15], exact=x))

    cardinality_bob = x_a * x_b * x_c * x_d

    # return non-local quantum numbers individually to facilitate m_\gamma constraint

    return cardinality_bob, k_1_, w_2_, nu_0_, nu_1_, nu_4_, nu_5_, k_1_ + w_2_ + nu_0_ + nu_1_ + nu_4_ + nu_5_


# Calculates the number of configurations of the symbols A and B in Alice's event (G_{a1})
def config_alice_ab(count_array_):

    A_a1 = count_array_[0] + count_array_[1] + count_array_[2] + count_array_[3]
    B_a1 = count_array_[4] + count_array_[5] + count_array_[6] + count_array_[7]

    return mf(A_a1 + B_a1)/(mf(A_a1)*mf(B_a1))


# Calculates the number of configurations of the symbols A and B in Bob's event (G_{b2})
def config_bob_ab(count_array_):

    A_b2 = count_array_[0] + count_array_[4] + count_array_[8] + count_array_[12]
    B_b2 = count_array_[1] + count_array_[5] + count_array_[9] + count_array_[13]

    return mf(A_b2 + B_b2) / (mf(A_b2) * mf(B_b2))


# The elementary path counting procedure (|L^a||L^b|)
def path_counter(n_, j_1a_, m_1a_, l_1a_, j_b2_, m_b2_, l_b2_, alpha_map_, beta_map_, gamma_map_,  m_gamma_, coefficient_array_):
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

    # sum over stand-ins for non-local quantum numbers
    ac_range_ = np.arange(0, j_b2_ + m_b2_ + 1)
    ad_range_ = np.arange(0, j_b2_ - m_b2_ + 1)
    cc_range_ = np.arange(0, min(j_1a_ - m_1a_, j_b2_ + m_b2_) + 1)
    ca_range_ = np.arange(0, j_1a_ - m_1a_ + 1)
    dd_range_ = np.arange(0, min(j_1a_ + m_1a_, j_b2_ - m_b2_) + 1)
    dc_range_ = np.arange(0, min(j_1a_ + m_1a_, j_b2_ + m_b2_) + 1)

    # Define array to store quantum numbers (to be used with the coefficient array to calculate counts)
    q_number_array = np.ndarray(shape=(16, 16))

    # Define the count array, which stores the counts associated with every complete set of quantum numbers
    count_array = []
  
    # sum over non-local quantum numbers
    for ac in range(len(ac_range_)):
        for ad in range(len(ad_range_)):
            for cc in range(len(cc_range_)):
                for ca in range(len(ca_range_)):
                    for dd in range(len(dd_range_)):
                        for dc in range(len(dc_range_)):
                            # Fill the array with the chosen values for the complete set of quantum numbers
                            q_number_array[:] = n_, j_1a_, m_1a_, l_1a_, j_b2_, m_b2_, l_b2_, alpha_map_, beta_map_, gamma_map_, ac_range_[ac], ad_range_[ad], cc_range_[cc], ca_range_[ca], dd_range_[dd], dc_range_[dc]

                            # Calculate the associated counts
                            count_array.append(np.sum(q_number_array * coefficient_array_, 1))

    # Make sure the count array is valid
    count_array = np.asarray(count_array)
    count_array_groomed = []

    # Identify situations in which some error has prevented a single set of counts from being defined.
    if len(count_array) > 0:
        # Identify the invalid entries in the count array.
        # Inner condition 1 requires that each count is greater than 0
        # Inner condition 2 requires that each count is an integer
        # If both conditions are met for all 16 counts, the outer condition is met.
        mask = np.where(np.sum(np.where((count_array >= 0) & (count_array % 1 == 0), 1, 0), 1) == 16, 1, 0)

        # Fill a new array with only the sets of counts that were determined to be valid.
        for t in range(len(mask)):
            if mask[t] == 1:
                count_array_groomed.append(count_array[t])

        # Identify situations in which there are no valid sets of counts

        if len(count_array_groomed) > 0:
            count_array_groomed = np.asarray(count_array_groomed)

            # Use the valid sets of counts to calculate the cardinality of Alice's and Bob's sub-ensembles
            phi_alice_ = card_alice_vec(count_array_groomed)
            phi_bob_ = card_bob_vec(count_array_groomed)

            # Now that we know we've found a valid set of quantum numbers, get the number of configurations of the
            # symbols A and B in Alice's and Bob's events.
            alice_config = config_alice_ab(count_array_groomed[0])
            bob_config = config_bob_ab(count_array_groomed[0])

            # Calculate the cardinalities of Alice's and Bob's local state spaces
            for v in range(len(count_array_groomed)):
                # Apply m_gamma constraint
                check_a = m_gamma_ == n_/4 + m_1a_/2 - l_1a_/2 - 2*(2*phi_alice_[2][v]+phi_alice_[3][v]+phi_alice_[4][v]-phi_alice_[5][v]-phi_alice_[6][v])

                if check_a:
                    alice_local_state_space += phi_alice_[0][v]*(-1)**(abs(phi_alice_[7][0]-phi_alice_[7][v]))
                    bob_local_state_space += phi_bob_[0][v]*(-1)**(abs(phi_bob_[7][0]-phi_bob_[7][v]))

    # return the total number of ordered pairs of states in Alice's and Bob's product space (after interference).
    # scale by the number of configurations of the symbols A and B in Alice's and Bob's event
    return alice_local_state_space*bob_local_state_space*alice_config*bob_config


# Calculates the probability as a function of theta (alpha_map)
def prob(n_, j_1a_, j_b2_, alpha_map_, gamma_map_,  m_gamma_, coefficient_array_):
    m_1a_range = np.arange(-j_1a_, j_1a_ + 1)
    m_b2_range = np.arange(-j_b2_, j_b2_ + 1)
    beta_range = np.arange(abs(alpha_map_ - gamma_map_), alpha_map_ + gamma_map_ + 1, 1)

    label_array_ = []
    upsilon_array_ = np.zeros(shape=(len(m_1a_range)*len(m_b2_range)))
    index_ = 0

    # sum over random variables
    for a in range(len(m_1a_range)):
        for b in range(len(m_b2_range)):
            label_array_.append(str(m_1a_range[a])+"  "+str(m_b2_range[b]))

            l_1a_range_ = np.arange(-(n_ / 2 - j_1a_), (n_ / 2 - j_1a_) + 1, 1)
            l_b2_range_ = np.arange(-(n_ / 2 - j_b2_), (n_ / 2 - j_b2_) + 1, 1)

            # sum over local nuisance variables
            for e in range(len(beta_range)):
                for c in range(len(l_1a_range_)):
                    for d in range(len(l_b2_range_)):
                        upsilon_array_[index_] += path_counter(n_, j_1a_, m_1a_range[a], l_1a_range_[c], j_b2_, m_b2_range[b], l_b2_range_[d], alpha_map_, beta_range[e], gamma_map_,  m_gamma_, coefficient_array_)
            index_ += 1

    probability = upsilon_array_/np.sum(upsilon_array_)
    return probability, label_array_


# Calculates the probability distribution as a function of theta_ab in the new model
def prob_rot(n_, j_1a_, j_b2_, gamma_map_, m_gamma_, coefficient_array_, x_):
    # Define arrays that will store data
    probability_data = []
    theta_data = []

    # Initialize alpha_map at 0
    # alpha_map is proportional to theta_ab
    alpha_map_ = 0
    labels = prob(n_, j_1a_, j_b2_, alpha_map_, gamma_map_,  m_gamma_, coefficient_array_)[1]
    # Sum over all values of b_map, from 0 to n
    while alpha_map_ <= n_:
        # Print the current value of alpha_map/n so the user can track progress
        print("alpha_map_= ", alpha_map_, "/", n)

        theta = math.pi * alpha_map_ / n_ - x_ * math.sin(2 * math.pi * alpha_map_ / n_)

        probability_data.append(prob(n_, j_1a_, j_b2_, alpha_map_, gamma_map_,  m_gamma_, coefficient_array_)[0])
        theta_data.append(theta)

        alpha_map_ += 1
    return theta_data, probability_data, labels


# Calculates the expectation values used for the CHSH calculation
def expectation(n_, alpha_map_, coefficient_array_):
    # Define Bell test conditioning variables

    j_1a_ = 1 / 2
    j_b2_ = 1 / 2
    gamma_map_ = 0
    m_gamma_ = 0

    temp = prob(n_, j_1a_, j_b2_, alpha_map_, gamma_map_,  m_gamma_, coefficient_array_)

    return temp[0][3] + temp[0][0] - temp[0][1] - temp[0][2]


# Maps theta (or theta_prime) to alpha
def get_alpha(n_, theta_, x_):
    # init variables for alpha run
    alpha_ = 0
    r_ = math.pi*alpha_/n_ - x_*math.sin(2*math.pi*alpha_/n_)
    r_array = [r_]

    # Convert theta to rads
    theta_rad_ = theta_*math.pi/180

    while r_ < theta_rad_:
        alpha_ += 1
        r_ = (math.pi*alpha_)/n_ - x_*math.sin(2*math.pi*alpha_/n_)
        r_array.append(r_)

    if abs(theta_rad_ - r_array[alpha_-1]) < abs(theta_rad_ - r_array[alpha_]):
        alpha_ = alpha_ - 1

    return alpha_


# Calculates the CHSH test statistic
def chsh_calc(n_, angle_a_, angle_ap_, angle_b_, angle_bp_, x_, coefficient_array_):
    x_ab = expectation(n_, get_alpha(n_, abs(angle_a_ - angle_b_), x_), coefficient_array_)
    x_abp = expectation(n_, get_alpha(n_, abs(angle_a_ - angle_bp_), x_), coefficient_array_)
    x_apb = expectation(n_, get_alpha(n_, abs(angle_ap_ - angle_b_), x_), coefficient_array_)
    x_apbp = expectation(n_, get_alpha(n_, abs(angle_ap_ - angle_bp_), x_), coefficient_array_)

    return abs(x_ab - x_abp + x_apb + x_apbp)


# Get results for a single choice of theta
# data = prob(n, j_1a, j_b2, alpha_map, 2*j_gamma,  m_gamma, coefficient_array)
# for y in range(len(data[0])):
#     print(data[1][y], ": ", data[0][y])


# Get results for a full range of theta (can be slow for large n and or j)
# rot_data = prob_rot(n, j_1a, j_b2, 2 * j_gamma, m_gamma, coefficient_array, x)
# np.save('spin one half 11.npy', rot_data)
# plt.plot(rot_data[0], rot_data[1])
# plt.legend(rot_data[2])
# plt.show()


# Perform CHSH calculation
chsh = chsh_calc(n,  0, 90, 45, 135, x, coefficient_array)
print("chsh: ", abs(chsh))

# Get results of CHSH calculation for a range of n. 
# chsh_n_data = []
# n_index = 1
# while n_index < n + 1:
#     chsh_n_data.append([n_index, chsh_calc(n_index, 0, 90, 45, 135, x, coefficient_array)])
#     print(n_index)
#     n_index += 1

# np.save('n_max=100 CHSH Data prime.npy', chsh_n_data)
# plt.scatter([arr[0] for arr in chsh_n_data], [arr[1] for arr in chsh_n_data])
# plt.show()

print(time.clock())
