import math

class Util:
    @classmethod
    def cal_dist(cls, p1:tuple, p2:tuple) -> float: #  L1 distanc
        dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        return float(dist)

if __name__ == '__main__':
    print(Util.cal_dist((0,0),(7.5,9.9)))
    print(Util.cal_dist((0,0),(7,9)))
    if math.isclose(Util.cal_dist((0,0),(7,9.0)), 16):
        print('equal to 16')
    if math.isclose(Util.cal_dist((0,0),(7.5,9.9)), 17.4):
        print('equal to 17.4')

