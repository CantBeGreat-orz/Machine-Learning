import numpy as np
import matplotlib.pyplot as plt
import random

class Kmeans:
    def __init__(self, cm_light, cm_dark):
        '''initialize the param'''
        self.cm_light = cm_light
        self.cm_dark = cm_dark

    def load_data(self, x, label, k):
        '''initialize the data'''
        self.x = x
        self.label = label
        self.k = k

    def init_center(self, rand , plus):
        if plus:
            '''method of kmeans++'''
            centers = [self.x[random.randint(0,self.x.shape[0]-1)]]
            for i in range(1,self.k):
                dis = np.zeros((self.x.shape[0], i))
                for j in range(0,i):
                    tmp = self.x - centers[j]
                    dis[:, j] = np.diagonal(np.matmul(tmp, tmp.T))
                dis = np.max(dis, axis = 1)
                '''RouletteWheelSelection'''
                odd = dis / np.sum(dis)
                r = random.random()
                p = 0
                for j in range(self.x.shape[0]):
                    p = p + odd[j]
                    if p > r:
                        centers.append(self.x[j])
                        break;
            centers = np.array(centers)
        elif rand == False:
            '''randomly generate the param within the range'''
            ids = np.arange(self.x.shape[0], dtype=int)
            np.random.shuffle(ids)
            centers = np.array([self.x[i] for i in ids[:self.k]])
        else:
            '''randomly choose centers'''
            x_max, x_min = np.max(self.x,axis=0),np.min(self.x,axis=0)
            #print(x_max,x_min)
            det = x_max -x_min
            #print(det)
            centers = np.random.rand(self.k,self.x.shape[1])*det+x_min
        return centers

    def run(self, iter_times, animation = True, rand = True, plus = False):
        '''visually run the procedure'''
        if(animation):
            fig = plt.figure(figsize=(9, 4))
            ax_loss = plt.subplot(1, 3, 1)
            ax_loss.set_title('WCSS Loss')
            ax_result = plt.subplot(1, 3, 2)
            ax_real = plt.subplot(1, 3, 3)
            ax_real.set_title('Real Classification')
            ax_real.scatter(self.x[:, 0], self.x[:, 1], c=self.label, edgecolor='k',
                            linewidths=0.1, alpha=0.9, s=30, cmap=self.cm_light)
            # mng = plt.get_current_fig_manager()
            # mng.resize(*mng.window.maxsize())
            # fig.canvas.manager.window.wm_geometry('+0+0')
        centers = self.init_center(rand, plus)
        for i in range(iter_times):
            print(centers)
            loss = 0
            dis = np.zeros((self.x.shape[0], self.k))
            for j in range(self.k):
                tmp = self.x - centers[j]
                dis[:, j] = np.diagonal(np.matmul(tmp, tmp.T))
            classification = np.argmin(dis, axis=1)
            new_centers = np.zeros(centers.shape)
            for j in range(self.x.shape[0]):
                new_centers[classification[j]] += self.x[j]
                loss += dis[j, classification[j]]
            for j in range(self.k):
                cnt = np.sum((classification == j))
                new_centers[j] /= cnt
            centers = new_centers
            if i > 0:
                if animation:
                    ax_loss.plot([i, i+1], [loss_last, loss], 'b-')
                if loss == loss_last:
                    break
            print('step: ', i+1, 'loss: ', loss)
            loss_last = loss
            if(animation):
                ax_result.cla()
                ax_result.set_title('Step '+str(i+1))
                ax_result.scatter(self.x[:, 0], self.x[:, 1], c=classification,
                                edgecolor='k', linewidths=0.1, alpha=0.9, s=30, cmap=self.cm_light)
                ax_result.scatter(centers[:, 0], centers[:, 1], c=[i for i in range(
                    centers.shape[0])], cmap=self.cm_dark, edgecolor='k', linewidths=1, marker='o', s=40)
                plt.pause(0.001)
        if(animation):
            plt.show()
        return centers
