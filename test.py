import numpy as np
from matplotlib import pyplot as plt
import os


def get_cell(M, x, y):
    # Loop when on boundary
    sz = np.size(M, axis=0)
    while x < 0:
        x = x + sz
    while y < 0:
        y = y + sz
    while x >= sz:
        x = x - sz
    while y >= sz:
        y = y - sz
    return M[x, y, :]


def get_max_specie(c, threshold):
    m = np.max(c)
    if m > threshold:
        return np.where(c == m)[0]
    return -1


def get_img(M, colors, threshold):
    sz = np.size(M, axis=0)
    img = np.ones([sz, sz, 3])
    for i in range(sz):
        for j in range(sz):
            s = get_max_specie(M[i, j, :], threshold)
            if s != -1:
                img[i, j, :] = colors[s, :]
    return img


n = 5
sz = 10


def get_complex(S):
    com = np.zeros([sz, sz])
    for i in range(n):
        com = com + S[:, :, i]
    m = np.maximum(np.max(com), 0.01)
    return com / m


def neighbors(i, j):
    Neighbors = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
    return [(i + dn[0], j + dn[1]) for dn in Neighbors]


def update(Th, S, T, u, W):
    # Th是theta
    global sz, n

    kla = np.array([0.1, 0.1, 10, 10, 100]) / 100
    alpha = np.array([0.23, 0.23, 0.17, 0.17, 0.14])
    r = np.array([200, 200, 20, 20, 5])
    tag = np.array([0.01, 0.01, 0, 0, 0])
    kuptake = np.array([0.1, 0.1, 5, 5, 20]) / 100
    theta_max = 0.42
    theta_wither = np.array([0.15, 0.15, 0.25, 0.25, 0.35])
    theta_T = np.array([0.1, 0.1, 0.06, 0.06, 0.04])
    k_root = np.array([0.1, 0.1, 5, 5, 20])
    kc4 = 0.001
    ti = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

    x_theta = np.exp(0.09130458981501738 / Th - 1) * 30.900216387798018
    x_theta = np.minimum(x_theta, 90)
    print('x_theta=', x_theta[0, 0])
    x = x_theta / 49.5
    k = np.ones([sz, sz])
    for i in range(n):
        k = k - S[:, :, i] * k_root[i]
    k = k * kc4
    Wi = (1 - x * x / (49.5 * 49.5)) * Th
    print('Wi=', Wi[0, 0])

    tmp = np.ones([sz, sz])
    for i in range(n):
        tmp = tmp * (2 * S[:, :, i] * kla[i] - alpha[i])
    Te = (1 - tmp) * T
    print('Te=', Te[0, 0])
    ue = np.ones([sz, sz])
    for i in range(n):
        ue = ue - S[:, :, i] * tag[i]
    ue = u * np.maximum(ue, 0)
    print('ue=', ue[0, 0])
    We = (-0.004580943228284413 * (Te + 1 / Te) + 0.041283648139215645 * (ue + 1))
    print('We=', We[0, 0])

    Wuptake = np.zeros([sz, sz])
    for i in range(n):
        Wuptake = Wuptake + S[:, :, i] * kuptake[i]
    print('Wuptake=', Wuptake[0, 0])

    dTh = (W - Wuptake - Wi - We)

    K = np.zeros([sz, sz, n])
    for i in range(n):
        K[:, :, i] = np.arctan((Th - theta_wither[i]) / theta_max)/1
    print('K=', K[0, 0])
    M = np.maximum(S * (1 + K), 1e-7)
    print('M=', M[0, 0])
    dS = np.zeros([sz, sz, n])
    for i in range(n):
        dS[:, :, i] = r[i] * S[:, :, i] * (1 - S[:, :, i] / M[:, :, i])

    # 追加散播dS
    for i in range(sz):
        for j in range(sz):
            for ni, nj in neighbors(i, j):
                # dS[i,j,:] = dS[i,j,:] + np.minimum(ti*get_cell(S,ni,nj), r)
                pass

    return np.maximum(np.minimum(Th + dTh * 0.1, 0.42), 0), np.maximum(S + dS * 0.1, 0)


if __name__ == "__main__":

    dist = np.array([10, 10, 4, 4, 0.1])

    S = np.random.random([sz, sz, n])  # lattice of species
    for i in range(n):
        S[:, :, i] = S[:, :, i] * dist[i]
    Th = np.random.random([sz, sz]) * 0.22 + 0.2  # lattice of water

    colors = np.random.random([n, 3])

    try:
        os.mkdir("out")
    except FileExistsError:
        pass
    T = 20
    u = 2
    W = 1
    tprint = print
    for i in range(365):
        R = np.random.random(3) - 0.5
        T = T + R[0]
        u = u + R[1] * 0.1
        W = W + R[2] * 0.01
        Wr = 0.1 if i < 365 / 4 else 0.3 if i < 365 / 2 else 0.1 if i < 365 * 3 / 4 else 0
        Wr = W * Wr
        if i % 30 == 0:
            print(T, u, Wr, '->', Th[0, 0], S[0, 0, :])
            # img = get_img(S, colors, 0.5)
            img = get_complex(S)
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="Greens")
            plt.subplot(1, 2, 2)
            plt.imshow(Th, cmap="Blues")
            plt.show()
            # plt.imsave("out/succeed{}.png".format(i), img)
        # print = lambda *args:None
        Th, S = update(Th, S, T, u, Wr)
        # print = tprint
