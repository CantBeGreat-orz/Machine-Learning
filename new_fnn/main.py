import os
from sklearn.model_selection import KFold
import numpy as np
from model.fnn import FNN


kf = KFold(n_splits=5)
iris_dic = {'Iris-setosa': 0, 'Iris-versicolor': 1,
            'Iris-virginica': 2}  # 标签映射


def load_data(data_path):
    '''载入数据'''
    x = np.loadtxt(data_path, delimiter=',', dtype=float,
                   ndmin=2, usecols=(0, 1, 2, 3))
    label = np.loadtxt(data_path, delimiter=',',
                       dtype=str, ndmin=2, usecols=(4))
    y = np.empty(shape=label.shape, dtype=int)
    for i in range(len(label)):
        y[i][0] = iris_dic[label[i][0]]
    return x, y


def shuffle_data(x, y):
    '''打乱原始数据，使得k折交叉验证时不会局限于某一标签的数据'''
    x_ret, y_ret = [], []
    idx0 = [i for i in range(0, 50)]
    idx1 = [i for i in range(50, 100)]
    idx2 = [i for i in range(100, 150)]
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)
    np.random.shuffle(idx2)
    idx = [idx0, idx1, idx2]
    for i in range(50):
        for j in range(3):
            x_ret.append(x[idx[j][i]])
            y_ret.append(y[idx[j][i]])
    # per = np.random.permutation(x.shape[0])
    return np.array(x_ret), np.array(y_ret)


def run_one_time(fnn, x, y, label_cnt, nodes_list, dirs):
    '''进行一次k折交叉验证'''
    test_accs = []
    train_accs = []
    losses = []
    i = 1
    for train_idx, test_idx in kf.split(x):
        # img_dir = dirs+"/["+",".join([str(x) for x in nodes_list])+"]_"+str(i)+".png" #节点数和层数的多轮测试的保存路径
        img_dir = dirs+"/"+str(fnn.alpha)+'_'+str(i)+".png"  # 测试不同节点数层数时的保存路径
        train_x, train_y = x[train_idx], y[train_idx]
        test_x, test_y = x[test_idx], y[test_idx]
        fnn.load_data(train_x, train_y, test_x, test_y,
                      label_cnt, nodes_list, img_dir)
        acc_test, acc_train, loss = fnn.run(
            decay=0.99, limit=0.1)  # 可以修改衰减率和衰减下限
        test_accs.append(acc_test)
        train_accs.append(acc_train)
        losses.append(loss)
        i += 1
    # 返回k次交叉验证的测试集准确率（list），训练集准确率（list），最后的随时函数值（list）
    return test_accs, train_accs, losses


if __name__ == '__main__':
    '''数据获取与打乱'''
    data_path = 'data/iris.txt'
    X, Y = load_data(data_path)
    np.set_printoptions(precision=5)
    x, y = shuffle_data(X, Y)
    # print(x,y)
    j = 'alpha(test)'  # 训练图片将存放在img/steps-n/round-j文件中
    '''学习率的多轮测试测试'''
    alpha_list = [0.1, 0.05, 0.02]
    steps = 300
    for h in range(len(alpha_list)):
        dirs = 'img/steps-' + str(steps) + '/round-' + str(j)
        if not os.path.exists(dirs):   # 如果不存在路径，则创建这个路径
            os.makedirs(dirs)
        fnn = FNN(alpha=alpha_list[h], epochs=1, steps=steps, mini_batch=0)
        test_accs, train_accs, losses = run_one_time(
            fnn, x, y, label_cnt=3, nodes_list=[10], dirs=dirs)
    '''节点数和层数的多轮测试'''
    # nodes = [[3], [5], [10], [15], [10, 10], [10, 20, 10]]
    # steps_list = [300, 400, 500, 600]
    # log_name = 'log_{}.txt'.format(str(j))#输出准确率等结果到文档
    # doc = open(log_name, 'w')
    # for h in range(len(steps_list)):
    #     dirs = 'img/steps-' + str(steps_list[h]) + '/round-' + str(j)
    #     if not os.path.exists(dirs):   # 如果不存在路径，则创建这个路径
    #         os.makedirs(dirs)
    #     accuracy = [[]for k in range(len(nodes))]
    #     loss = [[]for k in range(len(nodes))]
    #     print('max_iter:', steps_list[h])
    #     fnn = FNN(alpha=0.1, epochs=1, steps=steps_list[h], mini_batch=0)
    #     for k in range(len(nodes)):
    #         test_accs, train_accs, losses = run_one_time(
    #             fnn, x, y, label_cnt=3, nodes_list=nodes[k], dirs=dirs)
    #         aver_acc = np.mean(test_accs)
    #         loss[k].append(losses)
    #         # print('aver_acc: ', aver_acc)
    #         accuracy[k].append(aver_acc)
    #     accuracy_narray = np.array(accuracy)
    #     # accuracy_aver = np.mean(accuracy_narray,axis=1)
    #     print('accuracy', accuracy_narray)
    #     print('loss', loss)
    #     print('accuracy', accuracy_narray, file=doc)
    #     print('loss', loss, file=doc)
    #     # print('average accuracy: ', accuracy_aver)
    # doc.close()
