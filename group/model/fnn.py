import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess.normalize import min_max_normalization
from utils.preprocess.addbias import add_bias
from utils.function import sigmoid
from keras.utils.np_utils import to_categorical
from utils.function.predict import max_predict


class FNN:
    def __init__(self, alpha, epochs, steps, mini_batch, cm_light, cm_dark):
        '''initialize the param'''
        self.alpha = alpha
        self.cm_light = cm_light
        self.cm_dark = cm_dark
        self.epochs = epochs
        self.steps = steps
        self.mini_batch = mini_batch

    def load_data(self, x_train, y_train, x_test, y_test, label_cnt, nodes=5, layer_cnt=3, function=0):
        '''initialize and preprocess the data'''
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_onehot = to_categorical(y_train)
        self.x_test = x_test
        self.y_test = y_test
        #normalize & add_bias
        x_train_norm = min_max_normalization(x_train)
        x_test_norm = min_max_normalization(x_test)
        self.x_train_norm = add_bias(x_train_norm)
        self.x_test_norm = add_bias(x_test_norm)
        # self.theta = [self.generate_theta(nodes,self.x_train_norm.shape[1])]
        # for i in range(layer_cnt-3):
        #     self.theta.append(self.generate_theta(nodes,nodes+1))
        # self.theta.append(self.generate_theta(self.y_train_onehot.shape[1],nodes+1))
        self.theta1 = self.generate_theta(nodes, self.x_train_norm.shape[1])
        self.theta2 = self.generate_theta(
            self.y_train_onehot.shape[1], nodes+1)  # +1:add bias
        self.layer_cnt = layer_cnt
        if self.mini_batch <= 0:
            self.mini_batch = self.y_train.shape[0]

    def generate_theta(self, dim0, dim1):
        return np.random.rand(dim0, dim1)-0.5

    def run(self, decay=1, limit = 0.1):
        self.train(decay, limit)
        return self.test()

    def test(self):
        hypoth_test = self.hypothesize(
            add_bias(self.hypothesize(self.x_test_norm, self.theta1)), self.theta2)
        acc_train = self.get_acc(hypoth_test, self.y_test)
        print("acc: ", acc_train)
        return acc_train

    def train(self, decay, limit):
        '''run the learning procedure'''
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
            ids = np.arange(self.x_train_norm.shape[0], dtype=int)
            np.random.shuffle(ids)
            steps = int(self.x_train.shape[0]/self.mini_batch)
            left, right = 0, self.mini_batch
            for step in range(0,self.steps):
                loss, acc_train = self.grad_desc(ids)
                # left += self.mini_batch
                # right += self.mini_batch
                if iter > 0:
                    ax_loss.plot([iter, iter+1], [loss_last, loss], 'b-')
                loss_last = loss
                ax_acc.plot(iter, acc_train, 'b.', label='acc_train')
                if flag:
                    ax_acc.legend(loc='lower right', borderpad=0.2)
                    flag = False
                ax_result.cla()
                ax_result.set_title('Result')
                hypoth_grid = self.hypothesize(
                    add_bias(self.hypothesize(grid, self.theta1)), self.theta2)
                pred_grid = max_predict(hypoth_grid, xx.shape)
                ax_result.pcolormesh(xx, yy, pred_grid, cmap=self.cm_light)
                ax_result.scatter(self.x_train[:, 0], self.x_train[:, 1],
                                  c=self.y_train.flat, edgecolor='k', s=30, cmap=self.cm_dark)
                iter += 1
                if self.alpha > limit*alpha:
                    self.alpha *= decay
                print("epoch: ", i+1, " step: ", step+1,
                      " loss: ",  loss, ' acc_train:', acc_train)
                plt.pause(0.0001)
            self.alpha = alpha
        plt.show()
        self.alpha = alpha
        return acc_train

    def hypothesize(self, x, theta):
        '''get hypothesis'''
        product = np.matmul(x, theta.T)
        return sigmoid(product)

    # def predict(self, hypoth, y):
    #     return np.argmax(hypoth, axis=1).reshape(y.shape)

    def grad_desc(self,ids):
        hypoth_train = []
        x, theta = self.x_train_norm, self.theta1
        for i in range(self.layer_cnt-1):
            hypoth = self.hypothesize(x, theta)
            if i < self.layer_cnt-2:
                hypoth = add_bias(hypoth)
                hypoth_train.append(hypoth)
                x = hypoth
                theta = self.theta2
        y_hat = hypoth
        error_out = (y_hat - self.y_train_onehot)*y_hat*(1-y_hat)
        node = hypoth_train[-1]
        error_hidden = np.matmul(error_out, self.theta2)*node*(1-node)
        hypoth_train.pop()
        for i in ids:
            e = error_out[i].reshape(error_out.shape[1], 1)
            b = node[i].reshape(1, node.shape[1])
            self.theta2 -= self.alpha * np.matmul(e, b)
            e = error_hidden[i, 1:].reshape(error_hidden.shape[1]-1, 1)
            x = self.x_train_norm[i].reshape(1, self.x_train_norm.shape[1])
            self.theta1 -= self.alpha * np.matmul(e, x)
        loss = self.get_loss(y_hat)
        acc_train = self.get_acc(y_hat, self.y_train)
        return loss, acc_train

    def get_loss(self, hypoth):
        delta = hypoth - self.y_train_onehot
        return 0.5*np.sum(delta*delta)

    def get_acc(self, hypoth, y):
        pred = max_predict(hypoth, y.shape)
        acc = np.mean((pred == y).all(1))
        return acc
