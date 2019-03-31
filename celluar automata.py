import matplotlib.pyplot as plt
import numpy as np
import random

map = np.genfromtxt('a.csv', delimiter=',', skip_header=True, encoding='UTF-8')  # 优化前为a, 优化后为b
map = map.astype(int)

def wander(value, x, y, temp):
    U, D, R, L = 0, 0, 0, 0
    if temp[x+1][y] == 0:
        D = 1
    if temp[x-1][y] == 0:
        U = 1
    if temp[x][y+1] == 0:
        R = 1
    if temp[x][y-1] == 0:
        L = 1
    C = [D, U, R, L]
    if C == [0, 0, 0, 0]:
        return temp
    C = np.array(C)
    chlist = list(np.argwhere(C == 1))
    choice = random.sample(chlist, 1)[0][0]
    if choice == 0:                     # Down
        temp[x+1][y] = value
        temp[x][y] = 0
    if choice == 1:                     # Up
        temp[x - 1][y] = value
        temp[x][y] = 0
    if choice == 2:                     # R
        temp[x][y+1] = value
        temp[x][y] = 0
    if choice == 3:                     # L
        temp[x][y-1] = value
        temp[x][y] = 0
    return temp


def evolve(value, x, y, temp):
    if value == 0:
        temp[x, y] = 0
        return temp
    elif value == 1:
        choicewander = np.random.randint(0, 3)
        if choicewander != 0:
            if temp[x-1][y] == 0:               # 优先向上
                if x-1 == 1:                    # 碰到上界
                    temp[x-1][y] = 2            # 状态转移：窗口服务中
                    temp[x][y] = 0
                    return temp
                temp[x-1][y] = 1
                temp[x][y] = 0
                return temp
            right, left = 0, 0
            if temp[x][y + 1] == 0:
                right = 1
            if temp[x][y - 1] == 0:
                left = 1
            rl = [left, right]
            if rl == [0, 0]:                    # 左右都没
                if temp[x+1][y] == 0:           # 向下走
                    temp[x+1][y] = 1
                    temp[x][y] = 0
                return temp
            else:                               # 随机左右
                rl = np.array(rl)
                rllist = list(np.argwhere(rl == 1))
                choice = random.sample(rllist, 1)[0][0]
                if choice == 0:
                    if y - 1 == 1:  # 碰到左界
                        temp[x][y - 1] = 2  # 状态转移：窗口服务中
                        temp[x][y] = 0
                        return temp
                    temp[x][y-1] = value
                    temp[x][y] = 0
                    return temp
                if choice == 1:
                    if y + 1 == 48:  # 碰到右界
                        temp[x][y + 1] = 2  # 状态转移：窗口服务中
                        temp[x][y] = 0
                        return temp
                    temp[x][y + 1] = value
                    temp[x][y] = 0
                    return temp
            return temp
        if choicewander == 0:                           # 重写wander
            U, D, R, L = 0, 0, 0, 0
            if temp[x + 1][y] == 0:
                D = 1
            if temp[x - 1][y] == 0:
                U = 1
            if temp[x][y + 1] == 0:
                R = 1
            if temp[x][y - 1] == 0:
                L = 1
            C = [D, U, R, L]
            if C == [0, 0, 0, 0]:
                return temp
            C = np.array(C)
            chlist = list(np.argwhere(C == 1))
            choice = random.sample(chlist, 1)[0][0]
            if choice == 0:                         # Down
                temp[x + 1][y] = value
                temp[x][y] = 0
                return temp
            if choice == 1:                         # Up
                if x-1 == 1:                    # 碰到上界
                    temp[x-1][y] = 2            # 状态转移：窗口服务中
                    temp[x][y] = 0
                    return temp
                temp[x - 1][y] = value
                temp[x][y] = 0
                return temp
            if choice == 2:                         # R
                if y + 1 == 48:                     # 碰到右界
                    temp[x][y + 1] = 2              # 状态转移：窗口服务中
                    temp[x][y] = 0
                    return temp
                temp[x][y + 1] = 1
                temp[x][y] = 0
                return temp
            if choice == 3:                         # L
                if y - 1 == 1:                      # 碰到左界
                    temp[x][y - 1] = 2              # 状态转移：窗口服务中
                    temp[x][y] = 0
                    return temp
                temp[x][y - 1] = 1
                temp[x][y] = 0
                return temp
            return temp
    elif value >= 2 and value <= 10:
        temp[x][y] = temp[x][y]+1
        return temp
    elif value == 11:
        emptylist = np.argwhere(map == 20)
        if emptylist.shape[0] == 0:                         # 无空座位则随机走
            temp = wander(value=value, x=x, y=y, temp=temp)
            return temp
        currentPoint = np.array([x, y])
        mol = []
        for cal in range(0, emptylist.shape[0]):
            mol.append(np.linalg.norm(currentPoint - emptylist[cal]))       # 距离表
        mol = np.array(mol)
        shortest = mol.argsort()[0]                                         # 最短距离索引
        if mol[shortest] > 20:
            temp = wander(value=value, x=x, y=y, temp=temp)
            return temp
        temp[emptylist[shortest][0], emptylist[shortest][1]] = 21           # 瞬移至最短距离座位
        temp[x][y] = 0
        return temp
    elif value == 20:
        return temp
    elif value >= 21 and value <= 80:
        possible = np.random.randint(0, 2)
        if possible == 0:
            temp[x][y] = temp[x][y] + 1
            return temp
        else:
            return temp
    elif value == 81:
        temp[x][y] = 90
        return temp
    elif value >= 90 and value <= 93:
        temp[x][y] = temp[x][y] + 1
        return temp
    elif value == 94:
        temp[x][y] = 20
        return temp
    elif value == 100:
        return temp
    else:
        print("Error in values")


def switch(num):
    if num == 0:
        num = 1
        return num
    if num == 1:
        num = 0
        return num


def update(tempu):
    sequence = np.random.randint(0,2)
    if sequence == 0:
        for xi in range(24):
            for yj in range(49):
                valueu = tempu[xi, yj]
                tempu = evolve(valueu, xi, yj, tempu)
    if sequence == 1:
        for xi in range(24):
            for yj in reversed(range(49)):
                valueu = tempu[xi, yj]
                tempu = evolve(valueu, xi, yj, tempu)
    #if tempu[23, 23] == 0:
    #    tempu[23, 23] = 1
    if tempu[23, 24] == 0:
        tempu[23, 24] = 1
    if tempu[23, 25] == 0:
        tempu[23, 25] = 1
    #if tempu[23, 26] == 0:
    #    tempu[23, 26] = 1
    return tempu


def show(current):
    plt.matshow(current, cmap="flag")
    plt.show()


for times in range(0, 600):                         # output
    try:
        map = update(map)
        filename = repr(times) + ".png"
        fig = plt.matshow(map, cmap="flag")
        plt.savefig("1/"+filename, fmt='png')
        plt.cla()
        plt.clf()
        plt.close()
        plt.cla(fig)
        plt.close(fig)
    except TypeError:
        print("exception")
