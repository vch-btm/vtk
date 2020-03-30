import numpy as np
import platform
import os
import pylab as plt
import skfmm


def dtw(matrix):
    r, c = matrix.shape
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = matrix[i, j]
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])

    p, q, path = _traceback(D0)

    return D1[-1, -1] / sum(D1.shape), C, D1, p, q, path


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def _traceback(D):
    i, j = np.array(D.shape) - 2

    p, q = [i], [j]
    path = []
    path.insert(0, (i, j))


    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1

        path.insert(0, (i, j))
        p.insert(0, i)
        q.insert(0, j)

    return np.array(p), np.array(q), np.array(path)


def getBestPath(matrix):

    bestI = -1
    bestMat1 = 0
    bestMat2 = 0
    bestp = 0
    bestq = 0
    bestDist = np.inf
    bestPath = 0


    length = len(matrix[0])


    for i in range(1, length + 1):
        rolledMatrix = np.roll(matrix, i, axis=1)

        dist, mat1, mat2, p, q, path = dtw(rolledMatrix)

        if dist < bestDist:
            bestI = i
            bestMat1 = mat1
            bestMat2 = mat2
            bestp = p
            bestq = q
            bestDist = dist
            bestPath = path


    for i in range(len(bestPath)):
        bestPath[i, 1] = (bestPath[i, 1] - bestI + length) % length
        bestq[i] = (bestq[i] - bestI + length) % length

    return bestDist, bestMat1, bestp, bestq, bestPath




if platform.platform()[0] == "W":
    print("OS: win")
    pathIn = "c:/users/vch/desktop/results/withCushion/"
    pathOut = "c:/users/vch/desktop/results/withCushion/"


else:
    print("OS: not win")
    pathIn = "/home/horakv/Desktop/resultsRoot/withCushion/"
    pathOut = "/home/horakv/Desktop/resultsRoot/withCushion/"

if not os.path.exists(pathOut):
    os.makedirs(pathOut)


################################

factor = 3/4

for count in range(0, 23):

    tempMatrix = np.load('{}matrix_{}.npy'.format(pathIn, count))

    m, n = tempMatrix.shape
    div = int((m - 1) / (n - 1) * factor)

    tempMatrix = np.transpose(tempMatrix)
    tempMatrixLarger = np.zeros(((n-1) * div + 1, m))

    w = 1/(div + 1)


    for i in range(n-1):
        for j in range(div):
            if j == 0:
                tempMatrixLarger[i * (div) + j] = tempMatrix[i]
            else:
                tempMatrixLarger[i * div + j] = j * w * tempMatrix[i] + (1 - j * w) * tempMatrix[i + 1]


    tempMatrixLarger[(n-1) * div] = tempMatrix[n-1]
    tempMatrixLarger = np.transpose(tempMatrixLarger)

    np.save("{}matrix_{}_L".format(pathOut, count), tempMatrixLarger)

    tempMatrix = np.transpose(tempMatrix)

    #dist, mat1, mat2, p, q, path = dtw(tempMatrixLarger)

    dist, _, p, q, pathIn = getBestPath(tempMatrixLarger)

    fig, ax = plt.subplots()
    cax = ax.imshow(tempMatrixLarger, cmap=plt.cm.Reds)
    ax.set_aspect("auto")
    fig.colorbar(cax, orientation='vertical')
    plt.plot(q, p, "bo")
    plt.savefig("{}path_{}_L_{}.png".format(pathOut, count, factor))

    np.savetxt("{}path_{}_L_{}.txt".format(pathOut, count, factor), pathIn, fmt="%i %i") #, delimiter=',')