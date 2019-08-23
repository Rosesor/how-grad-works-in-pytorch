# inspire by https://blog.csdn.net/qq_35447659/article/details/83989668

class Cmytorch(object):
    def __init__(self):
        super(Cmytorch, self).__init__()

    def fun_test1(self):
        ##注意并未使用任何框架，纯手工打造，讲了pytorch线性拟合实现的原理
        ##演示torch一元线性拟合算法x * w + b
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['FangSong']  ##用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  ##用来正常显示负号
        torch.manual_seed(2019)

        ##准备要拟合的数据并将numpy数据转为pytorch类型数据
        # x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
        #                     [9.779], [6.182], [7.59], [2.167], [7.042],
        #                     [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
        #
        # y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
        #                     [3.366], [2.596], [2.53], [1.221], [2.827],
        #                     [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
        x_train = np.array([[3.3], [4.4], [5.5], [6.71]], dtype=np.float32)

        y_train = np.array([[1.7], [2.76], [2.09], [3.19]], dtype=np.float32)
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        ##定义参数 w 和 b，目的就是找出最合适的w和b来完成线性拟合
        ##构建线性拟合模型x * w + b
        w = torch.randn(1, requires_grad=True)  ##参与反向传导，也就是要参与训练
        b = torch.zeros(1, requires_grad=True)  ##参与反向传导，也就是要参与训练
        print(w)
        print(b)

        def linear_model(x):
            return torch.mul(x, w) + b

        ##定义loss的计算方式，这里使用均方误差
        def get_loss(my_pred, my_y_train):
            print('torch.mean((my_pred - my_y_train) ** 2)\n',torch.mean((my_pred - my_y_train) ** 2))
            return torch.mean((my_pred - my_y_train) ** 2)

        ##训练
        for e in range(10):
            pred = linear_model(x_train)  ##计算pred值
            loss = get_loss(pred, y_train)  ##计算loss值
            ##每一次都重新清空w,b的grad
            if w.grad:
                w.grad.zero_()
            if b.grad:
                b.grad.zero_()
            ##反向计算所有梯度，也就是让loss极小的所有训练参数梯度
            print(w)
            print(w.grad)
            print(b.grad)
            loss.backward()
            print(w.grad.data)
            print(b.grad.data)
            ##注意：lr这就是学习率的概念
            ##这里定大定小是比较关键的
            lr = 1e-2
            ##更新训练参数,这里使用计算出的梯度（偏导数）不断更新w,b的data部分
            ##理解为每次使用梯度的百分之一来更新训练参数
            w.data = w.data - lr * w.grad.data
            b.data = b.data - lr * b.grad.data
            if e % 20==0:
                print('epoch:{}, loss:{}, w:{}, b:{}'.format(e, loss.item(), w.item(), b.item()))

        ##训练完预测并画出效果
        y_ = linear_model(x_train)
        plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='原始数据')
        plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='拟合数据')
        plt.legend()
        plt.show()

        return


if __name__ == "__main__":
    ct = Cmytorch()
    ct.fun_test1()
