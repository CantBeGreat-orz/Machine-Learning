import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess.normalize import min_max_normalization
from utils.preprocess.addbias import add_bias
from utils.function import sigmoid
from keras.utils.np_utils import to_categorical
from utils.function.predict import max_predict


class FNN:
    def __init__(self, alpha, epochs, steps, mini_batch=0):
        '''initialize the param'''
        self.alpha = alpha
        self.epochs = epochs
        self.steps = steps
        self.mini_batch = mini_batch

    def load_data(self, x_train, y_train, x_test, y_test, label_cnt, nodes_list, img_dir):
        '''initialize and preprocess the data'''
        self.x_train = x_train
        self.y_train = y_train
        self.y_train_onehot = to_categorical(y_train)
        self.x_test = x_test
        self.y_test = y_test
        # normalize & add_bias
        x_train_norm = min_max_normalization(x_train)
        x_test_norm = min_max_normalization(x_test)
        self.x_train_norm = add_bias(x_train_norm)
        self.x_test_norm = add_bias(x_test_norm)
        #for image saving
        self.img_dir = img_dir
        # init the param
        self.layer_cnt = len(nodes_list)
        self.nodes_list = nodes_list
        self.theta = [self.generate_theta(
            nodes_list[0], self.x_train_norm.shape[1])]
        for i in range(self.layer_cnt-1):
            self.theta.append(self.generate_theta(
                nodes_list[i+1], nodes_list[i]+1))                   # +1:add bias
        self.theta.append(self.generate_theta(
            self.y_train_onehot.shape[1], nodes_list[-1]+1))  # +1:add bias
        # default use all the samples
        if self.mini_batch <= 0:
            self.mini_batch = self.y_train.shape[0]

    def generate_theta(self, dim0, dim1):
        '''random generate theta'''
        return np.random.rand(dim0, dim1)-0.5

    def run(self, decay=1, limit=0.1):
        '''start train and test'''
        acc_train, loss = self.train(decay, limit)
        return self.test(),acc_train, loss

    def test(self):
        test_input = self.x_test_norm
        for i in range(self.layer_cnt):
            test_input = add_bias(self.hypothesize(test_input, self.theta[i]))
        hypoth_test = self.hypothesize(test_input, self.theta[-1])
        acc_test = self.get_acc(hypoth_test, self.y_test)
        #print("acc: ", acc_test)
        return acc_test

    def train(self, decay, limit):
        '''run the learning procedure'''
        fig = plt.figure(figsize=(13, 4))
        ax_loss = plt.subplot(1, 2, 1)
        ax_loss.set_title('Loss')
        ax_acc = plt.subplot(1, 2, 2)
        ax_acc.set_title('Accuracy')
        ax_acc.set_ylim(0, 1.0)
        flag = True
        alpha = self.alpha
        iter = 0
        for i in range(self.epochs):
            ids = np.arange(self.x_train_norm.shape[0], dtype=int)
            np.random.shuffle(ids)
            steps = int(self.x_train.shape[0]/self.mini_batch)
            left, right = 0, self.mini_batch
            for step in range(0, self.steps):
                acc_test = self.test()
                loss, acc_train = self.grad_desc(ids)
                left += self.mini_batch
                right += self.mini_batch
                if iter > 0:
                    ax_loss.plot([iter, iter+1], [loss_last, loss], 'b-')
                loss_last = loss
                ax_acc.plot(iter, acc_train, 'b.', label='acc_train')
                ax_acc.plot(iter, acc_test, 'r.', label='acc_test')
                if flag:
                    ax_acc.legend(loc='lower right', borderpad=0.2)
                    flag = False
                iter += 1
                if self.alpha * decay > limit*alpha:
                    self.alpha *= decay
                print("epoch: ", i+1, " step: ", step+1, " loss: ",  loss,
                      ' acc_train:', acc_train, ' acc_test:', acc_test)
                # plt.pause(0.0001) #animation
            self.alpha = alpha
        # plt.show() #animation
        plt.savefig(fname=self.img_dir) #save image  
        # fig.clear()
        plt.close('all')
        self.alpha = alpha
        return acc_train,loss

    def hypothesize(self, x, theta):
        '''get hypothesis'''
        product = np.matmul(x, theta.T)
        return sigmoid(product)

    def grad_desc(self, ids):
        # Forward propagation
        net_input = [self.x_train_norm]
        for i in range(self.layer_cnt+1):
            theta = self.theta[i]
            hypoth = self.hypothesize(net_input[-1], theta)
            if i < self.layer_cnt:
                hypoth = add_bias(hypoth)
                net_input.append(hypoth)
        # calculate error
        y_hat = hypoth
        error = [(y_hat - self.y_train_onehot)*y_hat*(1-y_hat)]
        for i in range(len(net_input)-1, 0, -1):
            node = net_input[i]
            if i == len(net_input)-1:
                error.append(np.matmul(error[-1], self.theta[i])*node*(1-node))
            else:
                error.append(
                    np.matmul(error[-1][:, 1:], self.theta[i])*node*(1-node))
        # back propagation
        for i in ids:
            for j in range(self.layer_cnt):
                error_idx = self.layer_cnt-j
                e = error[error_idx][i, 1:].reshape(
                    error[error_idx].shape[1] - 1, 1)
                ni = net_input[j][i].reshape(1, net_input[j].shape[1])
                self.theta[j] -= self.alpha * np.matmul(e, ni)
            e = error[0][i].reshape(error[0].shape[1], 1)
            b = net_input[-1][i].reshape(1, net_input[-1].shape[1])
            self.theta[-1] -= self.alpha * np.matmul(e, b)
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
