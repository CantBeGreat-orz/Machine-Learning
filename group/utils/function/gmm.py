import numpy as np


class GMM:
    @staticmethod
    def get_probability(x, pi, u, sigma):
        k = u.shape[0]
        prob = np.zeros((x.shape[0], k))
        for j in range(k):
            x_tmp = x-u[j]
            tmp = np.matmul(x_tmp, np.linalg.inv(sigma[j]))
            tmp = np.sum(tmp * x_tmp, axis=1)
            prob[:, j] = pi[j] * np.exp(-0.5 * tmp)/(
                (np.linalg.det(sigma[j])**0.5)*(2*np.pi)**(x.shape[1]/2))
        return prob
    
    @staticmethod
    def get_param(x, gama, k):
        '''gama: one-hot label/probability array'''
        nk = np.sum(gama,axis=0)
        pi = nk / x.shape[0]
        u = np.zeros((k, x.shape[1]))
        for i in range(x.shape[0]):
            _x = x[i].reshape((1,x.shape[1]))
            g = gama[i].reshape((gama.shape[1],1))
            u +=  np.matmul(g,_x)
        u /= nk.reshape((k,1))
        sigma = np.zeros((k, x.shape[1], x.shape[1]))
        for _k in range(k):
            _x = x - u[_k]
            for i in range(x.shape[0]):
                x_tmp = _x[i]
                x_tmp = x_tmp.reshape((1, x.shape[1]))
                sigma[_k] += np.matmul(x_tmp.T, x_tmp)*gama[i,_k]
        sigma /= nk.reshape((k, 1, 1))
        return pi, u, sigma

