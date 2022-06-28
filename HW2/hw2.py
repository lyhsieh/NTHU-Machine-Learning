'''
NTHU EE Machine Learning HW2
Author: Lin Yung Hsieh
Student ID: 107061218
'''
import math
import argparse
import numpy as np
import pandas as pd
import scipy.stats


X1_MAX, X1_MIN, X2_MAX, X2_MIN = 0, 0, 0, 0


def calc_gaussian_basis(x1, x2, ui, uj, s1, s2):
    '''
    return the value of gaussian basis function
    '''
    return np.exp(-((x1 - ui) ** 2) / (2 * s1 ** 2) - ((x2 - uj) ** 2) / (2 * s2 ** 2))


def gen_phi(data, O1, O2):
    '''
    generate phi matrix
    '''
    GRE_score = data[:, 0]        # GRE_score is x1
    TOFEL_score = data[:, 1]      # TOFEL_score is x2
    research_exp = data[:, 2]     # research_exp is x3
    global X1_MAX
    global X1_MIN
    global X2_MAX
    global X2_MIN
    if len(GRE_score) == 300:     # we only need to set these args while training
        # print('Args set!')
        X1_MAX = np.max(GRE_score)
        X1_MIN = np.min(GRE_score)
        X2_MAX = np.max(TOFEL_score)
        X2_MIN = np.min(TOFEL_score)
    s1 = (X1_MAX - X1_MIN) / (O1 - 1)
    s2 = (X2_MAX - X2_MIN) / (O2 - 1)

    # construct phi table
    phi = np.zeros((len(GRE_score), O1 * O2 + 2))
    for row in range(len(GRE_score)):
        for i in range(1, O1 + 1):
            for j in range(1, O2 + 1):
                x1 = GRE_score[row]
                x2 = TOFEL_score[row]
                ui = s1 * (i - 1) + X1_MIN
                uj = s2 * (j - 1) + X2_MIN
                k = O2 * (i - 1) + j
                col = k - 1
                phi[row][col] = calc_gaussian_basis(x1=x1, x2=x2, ui=ui, uj=uj, s1=s1, s2=s2)
        phi[row][O1 * O2] = research_exp[row]
        phi[row][O1 * O2 + 1] = 1

    return phi

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    phi = gen_phi(data=train_data, O1=O1, O2=O2)
    chance_of_admit = train_data[:, 3]  # chance_of_admit is t

    alpha, beta = 0.5, 0.5
    I = np.identity(O1 * O2 + 2)
    sn = np.linalg.inv(alpha * I + beta * np.matmul(np.transpose(phi), phi))
    w = beta * np.matmul(sn, np.matmul(np.transpose(phi), chance_of_admit))

    phi = gen_phi(data=test_data_feature, O1=O1, O2=O2)

    y_BLRprediction = np.zeros(phi.shape[0])
    for i in range(phi.shape[0]):
        y_BLRprediction[i] = np.sum(np.dot(w, phi[i]))
    return y_BLRprediction

# do not change the name of this function
def MLR(train_data, test_data_feature, O1=5, O2=5):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    phi = gen_phi(data=train_data, O1=O1, O2=O2)
    chance_of_admit = train_data[:, 3]  # chance_of_admit is t

    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), \
        np.transpose(phi)), chance_of_admit)
    phi = gen_phi(data=test_data_feature, O1=O1, O2=O2)

    y_MLLSprediction = np.zeros(phi.shape[0])
    for i in range(phi.shape[0]):
        y_MLLSprediction[i] = np.sum(np.dot(w, phi[i]))

    return y_MLLSprediction

def CalMSE(data, prediction):
    '''
    calculating MSE
    '''
    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]
    return mean__squared_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=5)
    parser.add_argument('-O2', '--O_2', type=int, default=5)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2

    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)

    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(\
        e1=CalMSE(predict_BLR, data_test_label), \
        e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()
