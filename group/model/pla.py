import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess.normalize import min_max_normalization
from utils.preprocess.addbias import add_bias

class PLA:
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

    def run(self, decay = 0.9, limit=0.1):
        '''visually run the learning and prediction procedure'''
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
                product_grid, hypoth_grid = self.hypothesize(grid,xx)
                ax_result.pcolormesh(xx, yy, hypoth_grid, cmap=self.cm_light)
                ax_result.scatter(self.x_train[:, 0], self.x_train[:, 1],
                                  c=self.y_train.flat, edgecolor='k', s=30, cmap=self.cm_dark)
                iter += 1
                if self.alpha > limit * alpha:
                    self.alpha *= decay
                print("epoch: ", i+1, " step: ", step+1,
                      " loss: ",  loss, ' acc_train:', acc_train, ' acc_test:', acc_test)
                plt.pause(0.0001)
        self.alpha = alpha
        print("acc: ", acc_test)
        plt.show()
        return acc_test

    def hypothesize(self, x, y):
        '''get product & hypothesis'''
        product = np.matmul(x, self.theta)
        # get the max one as the hypothesis
        hypoth = np.argmax(product, axis=1).reshape(y.shape)
        return product, hypoth

    def grad_desc(self):
        '''stochastic gradient descent'''
        product_train, hypoth_train = self.hypothesize(
            self.x_train_norm, self.y_train)
        product_test, hypoth_test = self.hypothesize(
            self.x_test_norm, self.y_test)
        ids = np.arange(self.x_train_norm.shape[0], dtype=int)
        np.random.shuffle(ids)
        for i in ids[:self.mini_batch]:
            if hypoth_train[i, 0] != self.y_train[i, 0]:
                self.theta[:, self.y_train[i, 0]] += self.alpha * \
                    self.x_train_norm[i]
                self.theta[:, hypoth_train[i, 0]
                           ] -= self.alpha * self.x_train_norm[i]
        loss = self.get_loss(product_train, hypoth_train)
        acc_train, acc_test = self.get_acc(
            hypoth_train, self.y_train), self.get_acc(hypoth_test, self.y_test)
        return loss, acc_train, acc_test

    def get_loss(self, product, hypoth):
        loss = 0.0
        for i in range(self.x_train_norm.shape[0]):
            loss += product[i, hypoth[i, 0]] - product[i, self.y_train[i, 0]]
        return loss

    def get_acc(self, hypoth, y):
        acc = np.mean((hypoth == y).all(1))
        return acc
