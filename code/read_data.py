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
from collections import defaultdict as dd
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
    def __init__(self, k_favor, max_id=6000):
        self.k_favor = k_favor
        self.max_id = max_id

    def read_scratch(self):
        with open(data_dir + "twitter_combined.txt", 'r') as f:
            lines = map(lambda x: x.strip('\r\n').split(' '), f.readlines())
        user_dict = dd(list)
        count = 0
        for l in lines:
            source_u = int(l[0])
            target_u  = int(l[1])
            user_dict[source_u].append(target_u)
        for source_u in user_dict:
            if len(user_dict[source_u]) >= self.k_favor:
                count += 1
        print count
        with open(data_dir + 'twitter_5.json', 'w') as f:
            f.write(json.dumps(user_dict, indent=0))

    def popularity(self):
        with open(data_dir + "twitter_5.json",'r') as f:
            user_dict = json.loads(f.read())

        pop_count = dd(int)
        for user in user_dict:
            for target in user_dict[user]:
                pop_count[target] += 1

        pop_count = sorted(pop_count.items(), key=lambda x: x[1], reverse=True)
        pop_count = [p[0] for p in pop_count]
        pop_count = set(pop_count[:self.max_id])
        chosen_user = dd(list)
        count = 0
        for user in user_dict:
            if len(set(user_dict[user]).intersection(pop_count)) > self.k_favor:
                count += 1
                chosen_user[user] = list(set(user_dict[user]).intersection(pop_count))[:self.k_favor]
        print count
        return chosen_user

    def make_grid(self):
        chosen_user = self.popularity()
        target_dict = {}
        t_count = 0
        for user in chosen_user:
            for target in chosen_user[user]:
                if target not in target_dict:
                    target_dict[target] = t_count
                    t_count += 1
            chosen_user[user] = [target_dict[t] for t in chosen_user[user]]

        bit_array = np.full((len(chosen_user), t_count), False)
        for id_, user in enumerate(chosen_user.keys()):
            bit_array[id_, chosen_user[user]] = True

        sparse_bit_array = sparse.csr_matrix(bit_array, dtype=bool)
        print "{0} Finish Constructing Sparse Matrix, shape {1}, nonzeros {2}".format(datetime.now(),
                                                                                      sparse_bit_array.shape,
                                                                                      sparse_bit_array.nnz)
        return bit_array


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


# sg = SG(k_favor=10)
# sg.make_grid()
# sg.popularity()
# sg.make_grid()