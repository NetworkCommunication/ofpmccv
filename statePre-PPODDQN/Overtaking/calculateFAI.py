import numpy as np

k1 = 2.0
k2 = 2.0
k3 = 1.0
k4 = 1.0

def readData():
    vel = np.loadtxt("comparisonData/myself/speedControl.csv", delimiter=',')
    ttc = np.loadtxt("comparisonData/myself/TTC.csv", delimiter=',')
    lcn = np.loadtxt("comparisonData/myself/numberOfLaneChanges.csv", delimiter=',')
    return vel, ttc, lcn

def dealLCN(lcn):
    last_hundred = lcn
    average = sum(last_hundred) / len(last_hundred)
    # average = last_hundred
    return average

def dealVel(v):
    sum1 = 0
    n = len(v)
    max_v_diff = max([v[t + 1] - v[t] for t in range(len(v) - 1)])
    for t in range(len(v) - 1):
        if v[t + 1] == v[t]:
            sum1 += 1
        else:
            sum1 += abs(v[t + 1] - v[t]) / max_v_diff

    return -1 * sum1 / n

def dealTTC(ttc):
    max_TTC = max(ttc)
    sum2 = 0
    n = len(ttc)
    for t in range(len(ttc)):
        sum2 += ttc[t] / max_TTC
    return sum2 / n

def softmax():
    def normalize(x, min_val, max_val):
        normalized_x = (x - min_val) / (max_val - min_val)
        return normalized_x

    x = 2.5
    min_val = 0.0
    max_val = 3.0
    normalized_x = normalize(x, min_val, max_val)
    print(normalized_x)


if __name__ == '__main__':
    RMSE = [2.2, 3.9, 2.9, 6.4, 8.1, 4.3]
    vel, ttc, lcn = readData()
    finalVel = dealVel(vel)
    finalLCN = dealLCN(lcn)
    finalTTC = dealTTC(ttc)
    finalACC = sum(RMSE) / len(RMSE)

    FAI = k1 * finalVel + k2 * finalTTC + k3 / finalLCN + k4 * finalACC

    print("FAI:", FAI)

    # softmax()
