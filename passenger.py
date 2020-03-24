
class Passenger:
    def __init__(self, pick_up_point, drop_off_point):
        self.status = 'wait_pair' # wait_pair, wait_pick, picked_up, 'dropped'
        self.pick_up_point = pick_up_point
        self.drop_off_point = drop_off_point
        self.waiting_steps = 0

    def __repr__(self):
        return 'cls:' + type(self).__name__ + ', id:' + str(id(self))  + ', status: ' + self.status + ', pick_up_point:' + \
                str(self.pick_up_point) + ', drop_off_point:' + str(self.drop_off_point)

if __name__ == '__main__':
    p = Passenger((0,0), (1,0))
    print(p)
