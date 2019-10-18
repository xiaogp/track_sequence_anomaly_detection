import time

import numpy as np
from sklearn.neighbors import KDTree, BallTree
import faiss

POLY_COEF = [1.51025943e-09, 9.24903278e-06, -2.40148082e-06]


def data_preprocessing():
    points = {}
    with open(file_path, "r", encoding="utf8") as f:
        next(f)
        for index, line in enumerate(f.readlines()):
            line_list = line.strip().split(",")
            card_id = line_list[2]
            lng = float(line_list[3])
            lat = float(line_list[4])
            timestamp = time.localtime(int(line_list[1]) / 1000)
            tim = timestamp.tm_hour + timestamp.tm_min / 60

            # {index: {poi: [card_id, lng, lat, tim]]}
            points.setdefault(index, {"poi": {}})
            points[index]["poi"] = [card_id, tim, [lat, lng]]

    return points


class SearchMethod(object):
    def __init__(self, method, n_centers=10, search_centers=3):
        self.method = method
        self.n_centers = n_centers
        self.search_centers = search_centers
        self.model = None

    def fit(self, points):
        assert self.method in ["kdtree", "balltree", "faiss"], "method must in kdtree, balltree, faiss"
        if self.method == "faiss":
            n_dims = points.shape[1]
            model = faiss.IndexIVFFlat(faiss.IndexFlatL2(n_dims), n_dims, self.n_centers, faiss.METRIC_L2)
            model.train(points.astype(np.float32))
            model.add(points.astype(np.float32))
            model.nprobe = self.search_centers
        else:
            if self.method == "kdtree":
                model = KDTree(points)
            else:
                model = BallTree(points)

        self.model = model
        return self

    def top_search(self, point, top_n):
        if self.method == "faiss":
            return self.model.search(point.astype(np.float32), top_n)[1]
        return self.model.query(point, top_n, return_distance=False)

    def range_search(self, point, distance):
        if self.method == "faiss":
            return self.model.range_search(point.astype(np.float32), distance * distance)[2]
        return self.model.query_radius(point, distance)[0]


class STDBSCAN(object):

    def __init__(self, tim, geo, min_samples, method="kdtree"):
        self.tim = tim  # 时间范围
        self.geo = geo  # 地理距范围
        self.min_samples = min_samples  # 点数据量阈值
        self.method = method  # 近邻点检索方法
        self.geo_search_model = None
        self.tim_search_model = None

    def neighbors(self, points, point_index):
        target_tim = points[point_index]["poi"][1]
        target_geo = points[point_index]["poi"][2]
        max_tim = target_tim + self.tim
        geo_distance = self.geo
        
        # 使用np.polyfit拟合地理距离和欧式距离的二次函数表达式 
        predict_distances = POLY_COEF[0] * geo_distance * geo_distance + POLY_COEF[1] * geo_distance + POLY_COEF[2]
        geo_neighbors_list = self.geo_search_model.range_search(np.array([target_geo]), predict_distances)

        tim_neighbors_list = self.tim_search_model.range_search(np.array([[target_tim]]), self.tim)
        if target_tim + self.tim > 24:
            zero_radius_list = self.tim_search_model.range_search(np.array([[0]]), max_tim - 24)
            tim_neighbors_list = np.r_[tim_neighbors_list, zero_radius_list]

        neighbors_list = np.intersect1d(tim_neighbors_list, geo_neighbors_list)
        return neighbors_list

    def fit_predict(self, points):
        assert self.method in ["kdtree", "balltree", "faiss"], "method not in ['kdtree', 'balltree', 'faiss']"
        geo_search_model = SearchMethod(self.method)
        tim_search_model = SearchMethod(self.method)
        geo_array = np.array([points[i]["poi"][2] for i in range(len(points))])
        tim_array = np.array([points[i]["poi"][1] for i in range(len(points))]).reshape(-1, 1)
        geo_search_model.fit(geo_array)
        tim_search_model.fit(tim_array)
        self.geo_search_model = geo_search_model
        self.tim_search_model = tim_search_model

        label = 0  # 聚类类簇编号
        noise = -1  # 离群点类簇编号
        unknow = -2  # 还没有被访问的点编号

        for index in points:
            points[index]["label"] = unknow

        # {index: {poi: [card_id, lng, lat, tim]}, {label: -2}}
        stack = []
        for index, poi_label in points.items():
            if poi_label["label"] == unknow:
                neighbors_list = self.neighbors(points, index)

                # 判定是否是核心点
                if len(neighbors_list) < self.min_samples:
                    points[index]["label"] = noise

                else:
                    points[index]["label"] = label
                    for neighborhood in neighbors_list:
                        points[neighborhood]["label"] = label
                        stack.append(neighborhood)

                    while len(stack) > 0:
                        current_point_index = stack.pop()
                        current_neighbors_list = self.neighbors(points, current_point_index)
                        if len(current_neighbors_list) >= self.min_samples:
                            for neighborhood in current_neighbors_list:
                                if any([points[neighborhood]["label"] == unknow,
                                        points[neighborhood]["label"] == noise]):
                                    points[neighborhood]["label"] = label
                                    stack.append(neighborhood)
                    label += 1

        return points


if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("--tim", "-t", type=float, default=0.4,
                       help="[default %(default)s] neighbors time range")
    parse.add_argument("--geo", "-d", type=int, default=50,
                       help="[default %(default)s] neighbors geo distance range")
    parse.add_argument("--min_samples", "-ms", type=int, default=10,
                       help="[default %(default)s] neighbors range number of points")
    parse.add_argument("--method", "-md", type=str, default="kdtree",
                       help="[default %(default)s] search method for neighbors points")
    flags = parser.parse_args()

    file_path = "./todData/sample_data.csv"
    points = data_preprocessing()
    model = STDBSCAN(tim=flags.tim, geo=flags.geo, min_samples=flags.min_samples, method=flags.method)
    start = time.time()
    res = model.fit_predict(points)
    print(time.time() - start)

    with open("./dbscan_res.csv", "w", encoding="utf8") as f:
        f.write("card_id,hour,lat,lng,label\n")
        for i, j in res.items():
            label = j["label"]
            poi = j["poi"]
            f.write("{},{},{},{},{}\n".format(poi[0], poi[1], poi[2][0], poi[2][1], label))
