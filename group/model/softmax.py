import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess.normalize import min_max_normalization
from utils.preprocess.addbias import add_bias
from utils.function.predict import max_predict

class Softmax:
    def __init__(self, alpha, epochs, steps, mini_batch, cm_light, cm_dark):
        '''initialize the param'''
        self.alpha = alpha
        self.cm_light = cm_light
        self.cm_dark = cm_dark
        self.epochs = epochs
        self.steps = steps
        self.mini_batch = mini_batch

    def load_data(self, x_train, y_train, x_test, y_test, label_cnt):
        '''initialize and preprocess the data'''
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        #normalize & add_bias
        x_train_norm = min_max_normalization(x_train)
        x_test_norm = min_max_normalization(x_test)
        self.x_train_norm = add_bias(x_train_norm)
        self.x_test_norm = add_bias(x_test_norm)
        self.theta = np.random.rand(self.x_train_norm.shape[1], label_cnt)
        if self.mini_batch <= 0:
            self.mini_batch = self.y_train.shape[0]
    
    def run(self, decay = 1, limit = 0.1):
        '''run the learning and prediction procedure'''
        x_min, x_max = np.min(self.x_train, axis=0), np.max(
            self.x_train, axis=0)
        delta = x_max - x_min
        x_left_res, x_right_res = x_min[0]-delta[0]/20, x_max[0]+delta[0]/20
        y_left_res, y_right_res = x_min[1]-delta[1]/20, x_max[1]+delta[1]/20
        fig = plt.figure(figsize=(13, 4))
        ax_loss = plt.subplot(1, 3, 1)
        ax_loss.set_title('Loss')
        ax_acc = plt.subplot(1, 3, 2)
        ax_acc.set_title('Accuracy')
        ax_acc.set_ylim(0, 1.0)
        ax_result = plt.subplot(1, 3, 3)
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # fig.canvas.manager.window.wm_geometry('+0+0')
        xx, yy = np.meshgrid(np.linspace(x_left_res, x_right_res, 500), np.linspace(
            y_left_res, y_right_res, 500))
        grid_x = min_max_normalization(np.stack((xx.flat, yy.flat), axis=1))
        grid = add_bias(grid_x)
        # plt.ion()
        flag = True
        alpha = self.alpha
        iter = 0
        for i in range(self.epochs):
            for step in range(self.steps):
                #print(self.theta)
                loss, acc_train, acc_test = self.grad_desc()
                if iter > 0:
                    ax_loss.plot([iter, iter+1], [loss_last, loss], 'b-')
                loss_last = loss
                ax_acc.plot(iter, acc_train, 'b.', label='acc_train')
                ax_acc.plot(iter, acc_test, 'r.', label='acc_test')
                if flag:
                    ax_acc.legend(loc='lower right', borderpad=0.2)
                    flag = False
                ax_result.cla()
                ax_result.set_title('Result')
                hypoth_grid = self.hypothesize(grid)
                pred_grid = max_predict(hypoth_grid,xx.shape)
                ax_result.pcolormesh(xx, yy, pred_grid, cmap=self.cm_light)
                ax_result.scatter(self.x_train[:, 0], self.x_train[:, 1],
                                  c=self.y_train.flat, edgecolor='k', s=30, cmap=self.cm_dark)
                iter += 1
                if self.alpha > limit * alpha:
                    self.alpha *= decay
                print("epoch: ", i+1, " step: ", step+1,
                      " loss: ",  loss, ' acc_train:', acc_train, ' acc_test:', acc_test)
                plt.pause(0.0001)
            self.alpha = alpha
        # plt.ioff()
        print("acc: ", acc_test)
        plt.show()
        self.alpha = alpha
        return acc_test

    def hypothesize(self, x):
        '''get hypothesis'''
        product = np.matmul(x, self.theta)
        exp = np.exp(product)
        exp_sum = np.sum(exp,axis = 1,keepdims=True)
        hypoth = exp/exp_sum
        return hypoth

    # def predict(self,hypoth ,y):
    #     return np.argmax(hypoth, axis=1).reshape(y.shape)

    def grad_desc(self):
        hypoth_train = self.hypothesize(self.x_train_norm)
        hypoth_test = self.hypothesize(self.x_test_norm)
        ids = np.arange(self.x_train_norm.shape[0], dtype=int)
        np.random.shuffle(ids)
        for i in ids[:self.mini_batch]:
            y = self.y_train[i, 0]
            # if y>0:
            self.theta[:,y] += self.alpha *self.x_train_norm[i]
            for j in range(0,self.theta.shape[1]):
                self.theta[:,j] -= self.alpha *hypoth_train[i,j]*self.x_train_norm[i]
        loss = self.get_loss(hypoth_train)
        acc_train, acc_test = self.get_acc(
            hypoth_train, self.y_train), self.get_acc(hypoth_test, self.y_test)
        return loss, acc_train, acc_test

    def get_loss(self, hypoth):
        loss = 0.0
        for i in range(self.x_train_norm.shape[0]):
            loss += np.log(hypoth[i,self.y_train[i,0]])
        return -loss

    def get_acc(self, hypoth, y):
        pred = max_predict(hypoth, y.shape)
        acc = np.mean((pred == y).all(1))
        return acc