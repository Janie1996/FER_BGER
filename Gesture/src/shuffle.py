import numpy as np
import random



def shuffle(x_train, y_train):

    dim=x_train.shape
    num=x_train.shape[0]

    x = np.linspace(0, num - 1, num)
    random.shuffle(x)

    X_train = np.zeros(dim)
    Y_train = np.zeros((num,), dtype=np.int)

    for i in range(num):
        X_train[i] = x_train[int(x[i])]
        Y_train[i] = y_train[int(x[i])]

    print('finish train data shuffle')
    return X_train, Y_train


if __name__=="__main__":

    import numpy as np
    a=[1,2,3]                    # {1,2,3}
    a.append('hello')            # {1,2,3,'hello'}
    del a[0]                     # {2,3,'hello'}
    a1=np.array(a)
    b=np.ones(100)  #numpy
    b1=list(b)
    c=np.ones([2,3])   #numpy,2行3列
    d={'a':1,'b':2,'c':3}  #dict
    print(a)

