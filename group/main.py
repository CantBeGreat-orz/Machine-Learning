from random import random
import numpy as np
from sklearn.model_selection import KFold
from libsvm.svmutil import *
from libsvm.svm import *
import matplotlib.pyplot as plt
from model.pla import PLA
from model.softmax import Softmax
from model.kmeans import Kmeans
from model.fnn import FNN
from model.gmms import GMM_EM, GMM_Bayes
from colormap import cm_light as cl, cm_dark as cd

path_data = {3: 'data/GMM3.txt', 4: 'data/GMM4.txt',
             6: 'data/GMM6.txt', 8: 'data/GMM8.txt'}
kf = KFold(n_splits=5)
idx = 8
x = np.loadtxt(path_data[idx], skiprows=1,
               dtype=float, ndmin=2, usecols=(1, 2))
label = np.loadtxt(path_data[idx], skiprows=1,
                   dtype=int, ndmin=2, usecols=(0))


def get_model_pla(alpha, epochs, steps, mini_batch=0):
    pla = PLA(alpha, epochs, steps, mini_batch,
              cm_light=cl[idx], cm_dark=cd[idx])
    return pla

def get_model_softmax(alpha, epochs, steps, mini_batch=0):
    sfm = Softmax(alpha, epochs, steps, mini_batch,
                  cm_light=cl[idx], cm_dark=cd[idx])
    return sfm

def get_gmm_bayes():
    gb = GMM_Bayes(cm_light=cl[idx], cm_dark=cd[idx])
    return gb

def get_model_fnn(alpha, epochs, steps, mini_batch=0):
    fnn = FNN(alpha, epochs, steps, mini_batch,
              cm_light=cl[idx], cm_dark=cd[idx])
    return fnn

def get_kmeans():
    kmeans = Kmeans(cm_light=cl[idx], cm_dark=cd[idx])
    kmeans.load_data(x, label, k=idx)
    return kmeans

def get_gmm_em():
    gmm = GMM_EM(cm_light=cl[idx], cm_dark=cd[idx])
    gmm.load_data(x, label, k=idx)
    return gmm

def run_svm():
    param = svm_parameter('-s 0 -t 2 -c 1 -b 1')
    accs = []
    i = 1
    for train_idx, test_idx in kf.split(x):
        print('=====round ', i, '=====')
        train_x, train_y = x[train_idx], label[train_idx]
        test_x, test_y = x[test_idx], label[test_idx]
        prob = svm_problem(train_y.reshape(train_y.shape[0]), train_x)
        model = svm_train(prob, param)
        p_label, p_acc, p_val = svm_predict(test_y.reshape(test_y.shape[0]), test_x, model)
        x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
        delta = x_max - x_min
        x_left_res, x_right_res = x_min[0]-delta[0]/20, x_max[0]+delta[0]/20
        y_left_res, y_right_res = x_min[1]-delta[1]/20, x_max[1]+delta[1]/20
        plt.figure(figsize=(4, 4))
        ax_result = plt.subplot(1, 1, 1)
        xx, yy = np.meshgrid(np.linspace(x_left_res, x_right_res, 500), np.linspace(
            y_left_res, y_right_res, 500))
        grid_x = np.stack((xx.flat, yy.flat), axis=1)
        y_fake = np.zeros((250000,))
        _label, _acc, _val = svm_predict(y_fake, grid_x, model)
        ax_result.set_title('Result')
        ax_result.pcolormesh(xx, yy, np.array(
            _label).reshape(xx.shape), cmap=cl[idx])
        ax_result.scatter(x[:, 0], x[:, 1],
                          c=label.flat, edgecolor='k', s=30, cmap=cd[idx])
        plt.pause(0.0001)
        accs.append(p_acc[0])
        i += 1
    print('average accuracy: ', np.mean(accs))

def run_classification(model, decay=0.99, limit = 0.1):
    accs = []
    i = 1
    for train_idx, test_idx in kf.split(x):
        print('=====round ', i, '=====')
        train_x, train_y = x[train_idx], label[train_idx]
        test_x, test_y = x[test_idx], label[test_idx]
        model.load_data(train_x, train_y, test_x, test_y, idx)
        acc = model.run(decay, limit)
        accs.append(acc)
        i += 1
    print('average accuracy: ', np.mean(accs))

def run_cluster(model, iter_times, animation, rand, plus):
    model.run(iter_times, animation, rand, plus)

if __name__ == '__main__':
    '''choose model and run the task(classification/cluster)'''
    '''bayes based on gmm'''
    #m_classification = get_gmm_bayes()
    '''softmax regression'''
    #m_classification = get_model_softmax(alpha=0.1, epochs=1, steps=150, mini_batch=0)
    '''PLA model'''
    #m_classification = get_model_pla(alpha=0.1, epochs=1, steps=150, mini_batch=0)
    '''FNN model'''
    #m_classification = get_model_fnn(alpha=0.02, epochs=3, steps=300, mini_batch=0)
    '''k—_means'''
    m_cluster = get_kmeans() 
    '''em based on gmm '''
    #m_cluster = get_gmm_em() 
    #run_classification(m_classification, decay=0.99, limit = 0.5)
    run_cluster(m_cluster, iter_times=200, animation = True, rand = False, plus = False) 
    run_svm() #params can be modified in function body

