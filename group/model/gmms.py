import numpy as np
import matplotlib.pyplot as plt
from utils.function.gmm import GMM
from utils.function.predict import max_predict
from keras.utils.np_utils import to_categorical
from model.kmeans import Kmeans

class GMM_EM:
    def __init__(self, cm_light, cm_dark):
        '''initialize the param'''
        self.cm_light = cm_light
        self.cm_dark = cm_dark

    def load_data(self, x, label, k):
        '''initialize the data'''
        self.x = x
        self.label = label
        self.k = k

    def init_param(self, rand, plus):
        if rand:
            '''randomly generate the param within the range'''
            pi = np.random.rand(self.k)
            # pi = min_max_normalization(pi)
            u = np.random.rand(self.k, self.x.shape[1])*np.mean(self.x,axis=0)
            sigma = np.random.rand(self.k, self.x.shape[1], self.x.shape[1])
            return pi, u, sigma
        else:
            if plus:
                '''use kmeans'''
                kmeans = Kmeans(self.cm_light,self.cm_dark)
                kmeans.load_data(self.x, self.label, self.k)
                centers = kmeans.run(100, animation = False)
            else:
                '''randomly choose centers then calculate the params'''
                ids = np.arange(self.x.shape[0], dtype=int)
                np.random.shuffle(ids)
                centers = np.array([self.x[i] for i in ids[:self.k]])
            dis = np.zeros((self.x.shape[0], self.k))
            for j in range(self.k):
                tmp = self.x - centers[j]
                dis[:, j] = np.diagonal(np.matmul(tmp, tmp.T))
            classification = np.argmin(dis, axis=1).reshape((self.x.shape[0],1))
            gama = to_categorical(classification)
            return GMM.get_param(self.x, gama, self.k)
    

    def run(self, iter_times, animation = True, rand = True, plus = False):
        '''visually run the procedure'''
        fig = plt.figure(figsize=(9, 4))
        ax_loss = plt.subplot(1, 3, 1)
        ax_loss.set_title('Loss')
        ax_result = plt.subplot(1, 3, 2)
        ax_real = plt.subplot(1, 3, 3)
        ax_real.set_title('Real Classification')
        ax_real.scatter(self.x[:, 0], self.x[:, 1], c=self.label, edgecolor='k',
                        linewidths=0.1, alpha=0.9, s=30, cmap=self.cm_light)
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # fig.canvas.manager.window.wm_geometry('+0+0')
        pi,u,sigma = self.init_param(rand,plus)
        for k in range(self.k):
            sigma[k]+=np.identity(self.x.shape[1])
        for i in range(iter_times):
            '''e-step'''
            prob = GMM.get_probability(self.x,pi,u,sigma)
            classification = max_predict(prob,(self.x.shape[0],1))
            centers = np.zeros((self.k,self.x.shape[1]))
            for j in range(self.x.shape[0]):
                centers[classification[j]] += self.x[j]
            for j in range(self.k):
                cnt = np.sum((classification == j))
                centers[j] /= cnt
            prob_sum = np.sum(prob,axis=1,keepdims=True)
            loss = list(-np.sum(np.log(prob_sum),axis=0))[0]
            gama = prob/prob_sum
            '''m-step'''
            pi, u, sigma = GMM.get_param(self.x,gama,self.k)
            if i > 0:
                ax_loss.plot([i, i+1], [loss_last, loss], 'b-')
                if abs(loss - loss_last)<0.000001:
                    break
            print('step: ', i+1, 'loss: ', loss)
            loss_last = loss
            ax_result.cla()
            ax_result.set_title('Step '+str(i+1))
            ax_result.scatter(self.x[:, 0], self.x[:, 1], c=classification,
                              edgecolor='k', linewidths=0.1, alpha=0.9, s=30, cmap=self.cm_light)
            ax_result.scatter(centers[:, 0], centers[:, 1], c=[i for i in range(
                centers.shape[0])], cmap=self.cm_dark, edgecolor='k', linewidths=1, marker='o', s=40)
            if(animation):
                plt.pause(0.001)
        if(animation):
            plt.show()

class GMM_Bayes:
    def __init__(self, cm_light, cm_dark):
        '''initialize the param'''
        self.cm_light = cm_light
        self.cm_dark = cm_dark

    def load_data(self, x_train, y_train, x_test, y_test, label_cnt):
        '''initialize the data'''
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.label_cnt = label_cnt     

    def run(self, decay=1, limit = 1):
        gama = to_categorical(self.y_train)
        pi,u,sigma = GMM.get_param(self.x_train,gama,self.label_cnt)
        prob_train = GMM.get_probability(self.x_train,pi, u, sigma)
        prob_test= GMM.get_probability(self.x_test,pi, u, sigma)
        pred_train = max_predict(prob_train,self.y_train.shape)
        pred_test = max_predict(prob_test,self.y_test.shape)
        acc_train, acc_test = self.get_acc(
            pred_train, self.y_train), self.get_acc(pred_test, self.y_test)
        print("acc_train: ", acc_train, "acc_test: ", acc_test)
        x_min, x_max = np.min(self.x_train, axis=0), np.max(
            self.x_train, axis=0)
        delta = x_max - x_min
        x_left_res, x_right_res = x_min[0]-delta[0]/20, x_max[0]+delta[0]/20
        y_left_res, y_right_res = x_min[1]-delta[1]/20, x_max[1]+delta[1]/20
        fig = plt.figure(figsize=(4, 4))
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # fig.canvas.manager.window.wm_geometry('+0+0')
        ax_result = plt.subplot(1, 1, 1)
        xx, yy = np.meshgrid(np.linspace(x_left_res, x_right_res, 200), np.linspace(
            y_left_res, y_right_res, 200))
        grid_x = np.stack((xx.flat, yy.flat), axis=1)
        prob_grid = GMM.get_probability(grid_x,pi, u, sigma)
        pred_grid = max_predict(prob_grid,xx.shape)
        ax_result.set_title('Result')
        ax_result.pcolormesh(xx, yy, pred_grid, cmap=self.cm_light)
        ax_result.scatter(self.x_train[:, 0], self.x_train[:, 1],
                          c=self.y_train.flat, edgecolor='k', s=30, cmap=self.cm_dark)
        plt.show()
        print(pi)
        print(u)
        print(sigma)
        return acc_test

    # def get_u(self, sum_cate):
    #     u = np.zeros((self.label_cnt, self.x_train.shape[1]))
    #     for i in range(self.y_train.shape[0]):
    #         y = self.y_train[i, 0]
    #         u[y] += self.x_train[i]
    #     u = np.divide(u, sum_cate.reshape((self.label_cnt, 1)))
    #     return u

    # def get_sigma(self, u, sum_cate):
    #     sigma = np.zeros(
    #         (self.label_cnt, self.x_train.shape[1], self.x_train.shape[1]))
    #     for i in range(self.y_train.shape[0]):
    #         y = self.y_train[i, 0]
    #         x_tmp = self.x_train[i] - u[y]
    #         x_tmp = x_tmp.reshape((1, 2))
    #         sigma[y] += np.matmul(x_tmp.T, x_tmp)
    #     sigma = np.divide(sigma, sum_cate.reshape((self.label_cnt, 1, 1)))
    #     return sigma

    # def predict(self, pi, u, sigma, x, y):
    #     pred = np.zeros((x.shape[0], self.label_cnt))
    #     for i in range(x.shape[0]):
    #         for j in range(self.label_cnt):
    #             x_tmp = x[i]-u[j]
    #             x_tmp = x_tmp.reshape((1, x.shape[1]))
    #             tmp = np.matmul(x_tmp, np.linalg.inv(sigma[j]))
    #             pred[i][j] = pi[j] * np.exp(-0.5 * np.matmul(tmp, x_tmp.T))/(
    #                 (np.linalg.det(sigma[j])**0.5)*(2*np.pi)**(x.shape[1]/2))
    #     return np.argmax(pred, axis=1).reshape(y.shape)

    def predict(self, pi, u, sigma, x, y):
        pred = np.zeros((x.shape[0], self.label_cnt))
        for j in range(self.label_cnt):
            x_tmp = x-u[j]
            tmp = np.matmul(x_tmp, np.linalg.inv(sigma[j]))
            tmp = np.sum(tmp * x_tmp,axis=1)
            pred[:,j] = pi[j] * np.exp(-0.5 * tmp)/(
                (np.linalg.det(sigma[j])**0.5)*(2*np.pi)**(x.shape[1]/2))
        return pred, np.argmax(pred, axis=1).reshape(y.shape)

    def get_acc(self, pred, y):
        acc = np.mean((pred == y).all(1))
        return acc