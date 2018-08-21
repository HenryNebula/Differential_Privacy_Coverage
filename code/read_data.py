import numpy as np
import random
import json
import copy


from bokeh.io import output_file, show, export_png
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
from scipy.spatial import KDTree

from datetime import datetime
from scipy import sparse

MAX_LAT = 40.426
MIN_LAT = 39.414
MAX_LON = 117.119
MIN_LON = 115.686

pre_compute_dir = "../pre-compute/"
data_dir = "../data/"
output_dir = "../output/"

def init_pick(bit_array,candidate,random_start=5):
    best_choice = []
    max_cover = 0
    max_iter = 2000
    stop_iter = 500
    people = bit_array.shape[0]
    for _ in range(0, random_start):
        count = 0
        choice = random.sample(range(people), candidate)
        coverage = np.sum(np.sum(bit_array[choice, :], axis=0)>0)
        not_choice = list(set(range(people)).difference(set(choice)))
        for iter in range(0, max_iter):
            r = random.sample(choice, 1)[0]
            r_ = random.sample(not_choice,1)[0]
            try_choice = copy.copy(choice)
            try_choice.remove(r)
            try_choice.append(r_)
            new_cover = np.sum(np.sum(bit_array[try_choice, :], axis=0)>0)
            if new_cover > coverage:
                coverage = new_cover
                choice = try_choice
            else:
                count += 1
                if count > stop_iter:
                    print "Stop before max_iter: ", count
                    break
        if coverage > max_cover:
            best_choice = copy.copy(choice)
            max_cover = coverage
    print "max_coverage: ", max_cover
    return bit_array[best_choice, :]


def draw_map(lat, lon, title="MCS"):

    map_options = GMapOptions(lat=0.5*(MAX_LAT + MIN_LAT), lng=0.5*(MAX_LON + MIN_LON), map_type="roadmap", zoom=10)
    p = gmap("AIzaSyDwekyNM4fOE7byChkNKCgEXklUAn3FA6o", map_options, title="MCS")
    source = ColumnDataSource(
        data=dict(lat=lat,
                  lon=lon)
    )
    p.circle(x="lon", y="lat", size=5, fill_color="blue", fill_alpha=0.1, source=source)
    # export_png(p, output_dir + 'map/' + title + '.png')
    show(p)

# mobile crowdsourcing
class MCS():
    def __init__(self, k_favor, x_granu=0.02, y_granu=0.02):
        self.k_favor = k_favor
        self.x_granu = x_granu
        self.y_granu = y_granu

    def read_scratch(self, draw=False):
        # mesh a grid first
        x = np.arange(start=MIN_LAT, stop=MAX_LAT, step=self.x_granu)
        y = np.arange(start=MIN_LON, stop=MAX_LON, step=self.y_granu)

        xx, yy = np.meshgrid(x,y)
        xx, yy = xx.reshape(-1,1), yy.reshape(-1,1)
        grid = np.hstack((xx,yy))
        kdtree = KDTree(data=grid,)

        print "{0} Start Loading, x,y granularity:{1}".format(datetime.now(), (self.x_granu, self.y_granu))

        with open(data_dir + 'travel_plan', 'r') as f:
            l = f.readlines()
        loc_list = []
        for line in l:
            line = line.split(';')
            # user_visit = {ind: count}
            user_visit = {}
            line = [i.split(',') for i in line]
            # (lat, lon)
            line = [(float(i[1]),float(i[0])) for i in line]
            for loc in line:
                if loc[0] > MIN_LAT and loc[0] < MAX_LAT and loc[1] > MIN_LON and loc[1] < MAX_LON:
                    dis, ind = kdtree.query(np.array(loc))
                    if ind in user_visit.keys():
                        user_visit[ind] += 1
                    else:
                        user_visit[ind] = 1
            if len(user_visit.items()) >= self.k_favor:
                # most frequent visit is chosen
                ordered_visit = sorted(user_visit.items(), key=lambda x: x[1], reverse=True)
                inds, counts = zip(*ordered_visit)
                loc_list.extend(inds[0:self.k_favor])

        print "{0} Finish fitting to the grid, valid visit records: {1}".format(datetime.now(), len(loc_list))

        # visualize it
        if draw:
            whole_loc = list(kdtree.data[loc_list,:])
            lats, lons= zip(*whole_loc)
            draw_map(lats, lons)

        loc_list = np.array(loc_list)
        loc_list = loc_list.reshape(-1,self.k_favor)
        bit_array = np.full((loc_list.shape[0], grid.shape[0]), False)
        for row, indice in enumerate(loc_list):
            bit_array[row, indice] = True

        sparse_bit_array = sparse.csr_matrix(bit_array, dtype=bool)
        print "{0} Finish Constructing Sparse Matrix, shape {1}, nonzeros {2}".format(datetime.now(),
                                                                        sparse_bit_array.shape, sparse_bit_array.nnz)
        return bit_array


# social graph
class SG():
    def __init__(self, k_favor, max_id=1500):
        self.k_favor = k_favor
        self.max_id = max_id

    def read_scratch(self):
        with open(data_dir + "social_graph", 'r') as f:
            l = f.readlines()
        user_list = [list(set(item.strip('\r\n').split(' '))) for item in l]
        with open(data_dir + "social_graph.json",'w') as f_write:
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
        with open(data_dir + "social_dict.json",'w') as f_write:
            f_write.write(json.dumps(user_dict,indent=4))


    def popularity(self):
        with open(data_dir + "social_graph.json",'r') as f:
            user_list = json.loads(f.read())

        with open(data_dir + "social_dict.json",'r') as f_write:
            user_dict = json.loads(f_write.read())

        count_dict = {}
        for ul in user_list:
            for user in ul:
                if count_dict.has_key(user_dict[user]):
                    count_dict[user_dict[user]] += 1
                else:
                    count_dict[user_dict[user]] = 1

        popular = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        with open(data_dir + "popularity.json", 'w') as f:
            f.write(json.dumps(popular, indent=0))

    def make_grid(self):
        with open(data_dir + "social_graph.json",'r') as f:
            user_list = json.loads(f.read())
        with open(data_dir + "social_dict.json",'r') as f_write:
            user_dict = json.loads(f_write.read())
        with open(data_dir + "popularity.json",'r') as f_write:
            popular = json.loads(f_write.read())

        choose = []
        target_user = popular[0:self.max_id]
        id_dict = {}
        for i, t in enumerate(target_user):
            id_dict[t[0]] = i
        target_user = set([t[0] for t in target_user])


        for id_, user in enumerate(user_list):
            if len(user) < self.k_favor:
                continue
            user_id = map(lambda x: user_dict[x], user)
            user_id = set(user_id).intersection(target_user)
            user_id = list(user_id)
            # user_id = user_id[user_id < self.max_id]
            if len(user_id) >= self.k_favor:
                choose.append((id_, user_id[0:self.k_favor]))

        # bit_array = np.zeros((len(choose), self.max_id))
        bit_array = np.full((len(choose), self.max_id), False)
        for i, choice in enumerate(choose):
            for c in choice[1]:
                bit_array[i, id_dict[c]] = 1
        # return bit_array[random.sample(range(bit_array.shape[0]), self.people), :]
        print "Current recruited:{0}".format(bit_array.shape[0])
        return  bit_array
        # return init_pick(bit_array, self.people)


# contact graph
class CG():
    def __init__(self, k_favor, people, max_id=1000):
        self.k_favor = k_favor
        self.people = people
        # scaling paras for crowdsourcing dataset
        self.x_granu = 0.01
        self.y_granu = 0.01
        self.max_id = max_id

    def read_scratch(self):
        with open('travel_plan', 'r') as f:
            l = f.readlines()
        new_l = []
        for line in l:
            line = line.split(';')
            line = [i.split(',') for i in line]
            line = [(int(float(i[1])/ self.y_granu),
                     int(float(i[0])/ self.x_granu)) for i in line]
            line = set(line)
            new_l.append(list(line))

        contact_graph = []
        for id1, person1 in enumerate(new_l):
            neighbor = []
            for id2, person2 in enumerate(new_l):
                if id1 == id2:
                    continue
                if set.intersection(set(person1), set(person2)) != set():
                    neighbor.append(id2)
            print id1
            contact_graph.append((id1, neighbor))
        with open("Contact_graph.json",'w') as f:
            f.write(json.dumps(contact_graph))

    def make_grid(self):
        with open("Contact_graph.json","r") as f:
            l = json.loads(f.read())
        choose = []
        for man in l:
            id = man[0]
            neighbor = man[1]
            if len(neighbor) < self.k_favor:
                continue
            if neighbor[self.k_favor - 1] >= self.max_id:
                continue
            choose.append((id, neighbor[0:self.k_favor]))
        print "Now recruit people: ", len(choose)
        if len(choose) < self.people:
            print  "Ask for too many people, try reduce people or increase max_id"
            exit(1)
        bit_array = np.zeros((len(choose), self.max_id))
        for i,choice in enumerate(choose):
            bit_array[i, choice[1]] = 1
        return bit_array[0:self.people,:]


mcs = MCS(x_granu=0.0175, y_granu=0.0175,k_favor=5)
mcs.read_scratch(draw=False)

# sg = SG(k_favor=3)
# sg.popularity()
# sg.make_grid()