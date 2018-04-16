import numpy as np
import random
import json

# mobile crowdsourcing
class MCS():
    def __init__(self, k_favor, people, xrange=30, yrange=30):
        self.k_favor = k_favor
        self.people = people
        # scaling paras for crowdsourcing dataset
        self.x_granu = 0.01
        self.y_granu = 0.01
        self.xrange = xrange
        self.yrange = yrange

    def read_scratch(self):
        with open('travel_plan', 'r') as f:
            l = f.readlines()
        new_l = []
        for line in l:
            line = line.split(';')
            if len(line) >= self.k_favor:
                line = [i.split(',') for i in line]
                line = [(int(float(i[1])/ self.y_granu),
                         int(float(i[0])/ self.x_granu)) for i in line]
                line = list(set(line))
                if len(line) >= self.k_favor:
                    new_l.append(line[0:self.k_favor])
        X_coordinate = np.zeros((len(new_l), self.k_favor))
        Y_coordinate = np.zeros((len(new_l), self.k_favor))
        for id in range(0, len(new_l)):
            for pos_id in range(0, self.k_favor):
                X_coordinate[id, pos_id] = new_l[id][pos_id][0]
                Y_coordinate[id, pos_id] = new_l[id][pos_id][1]
        X_med = np.median(X_coordinate)
        Y_med = np.median(Y_coordinate)
        choice = []
        for id in range(0, len(new_l)):
            if np.all(X_coordinate[id,:] <= X_med + self.xrange) and\
                    np.all(X_coordinate[id,:] > X_med - self.xrange):
                if np.all(Y_coordinate[id,:] <= Y_med + self.yrange) and\
                        np.all(Y_coordinate[id,:] > Y_med - self.yrange):
                    choice.append(id)
        X = X_coordinate[choice,:]
        X = X - np.min(X)
        Y = Y_coordinate[choice,:]
        Y = Y - np.min(Y)
        X_range = 2 * self.xrange
        Y_range = 2 * self.yrange
        np.savez("XY",X=X, Y=Y,X_range=X_range, Y_range=Y_range)

    def make_grid(self):
        self.read_scratch()
        with np.load("XY.npz") as data:
            X = data["X"]
            Y = data["Y"]
            X_range = data["X_range"]
            Y_range = data["Y_range"]

        if self.people > X.shape[0]:
            print "Present recruit: ", X.shape[0]
            print "Require too many people, try reduce people, increase area, decrease k_favor"
            exit(1)
        # X_range = np.max(X) + 1
        # Y_range = np.max(Y) + 1
        self.plc_num = int(X_range * Y_range)
        flat_map = Y * X_range + X
        bit_array = np.zeros((X.shape[0], self.plc_num))
        for row in range(X.shape[0]):
            ones = [int(i) for i in list(flat_map[row,:])]
            bit_array[row, ones] = 1
        return bit_array[range(self.people),:]

# social graph
class SG():
    def __init__(self, k_favor, people, max_id=10000):
        self.k_favor = k_favor
        self.people = people
        # scaling paras for crowdsourcing dataset
        self.x_granu = 0.01
        self.y_granu = 0.01
        self.max_id = max_id

    def read_scratch(self):
        with open("social_graph", 'r') as f:
            l = f.readlines()
        user_list = [list(set(item.strip('\r\n').split(' '))) for item in l]
        with open("social_graph.json",'w') as f_write:
            f_write.write(json.dumps(user_list,indent=4))
        user_dict = {}
        count = 0
        for user in user_list:
            for id_ in user:
                if user_dict.has_key(id_):
                    continue
                else:
                    user_dict[id_] = count
                    count += 1
        with open("social_dict.json",'w') as f_write:
            f_write.write(json.dumps(user_dict,indent=4))

    def make_grid(self):
        with open("social_graph.json",'r') as f:
            user_list = json.loads(f.read())
        with open("social_dict.json",'r') as f_write:
            user_dict = json.loads(f_write.read())

        choose = []

        for id_, user in enumerate(user_list):
            if len(user) < self.k_favor:
                continue
            user_id = np.array(map(lambda x: user_dict[x], user))
            user_id = user_id[user_id < self.max_id]
            if user_id.shape[0] >= self.k_favor:
                choose.append((id_, user_id[0:self.k_favor]))
        print "Present recruit: ", len(choose)
        if self.people > len(choose):
            print "Require too many people, try reduce people or increase max_id"
            exit(1)
        bit_array = np.zeros((len(choose), self.max_id))
        for i in range(len(choose)):
            bit_array[i,choose[i][1]] = 1
        return bit_array[0:self.people,:]






# sg = SG(k_favor=5, people=2000, max_id=1000)
# # sg.read_scratch()
# sg.make_grid()