# -*- coding: utf-8 -*-

import time

from geopy.distance import great_circle


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
            points[index]["poi"] = [card_id, tim, (lat, lng)]

    return points


class st_dbscan(object):
    def __init__(self, tim, geo, min_samples):
        self.tim = tim  # 时间范围
        self.geo = geo  # 地理距范围
        self.min_samples = min_samples  # 点数据量阈值

    def calculate_geo_distance(self, lat_lng1, lat_lng2):
        distance = great_circle(lat_lng1, lat_lng2).meters
        return distance

    def neighbors(self, points, point_index):
        neighbors_list = []
        target_tim = points[point_index]["poi"][1]
        target_geo = points[point_index]["poi"][2]
        # 先判断时间
        max_tim = target_tim + self.tim if (target_tim + self.tim) < 24 else (target_tim + self.tim) - 24
        min_tim = target_tim - self.tim if (target_tim - self.tim) >= 0 else (target_tim - self.tim) + 24

        for index in points:
            if index == point_index:
                continue
            if max_tim > min_tim:
                if all([points[index]["poi"][1] <= max_tim,
                        points[index]["poi"][1] >= max_tim]):
                    distance = self.calculate_geo_distance(points[index]["poi"][2], target_geo)
                    if distance <= self.geo:
                        neighbors_list.append(index)
            else:
                if any([0 <= points[index]["poi"][1] <= max_tim,
                        min_tim <= points[index]["poi"][1] <= 24]):
                    distance = self.calculate_geo_distance(points[index]["poi"][2], target_geo)
                    if distance <= self.geo:
                        neighbors_list.append(index)

        return neighbors_list

    def fit_predict(self, points):
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
    file_path = "./todData/sample_data.csv"
    points = data_preprocessing()
    model = st_dbscan(tim=0.4, geo=300, min_samples=10)
    res = model.fit_predict(points)

    with open("./dbscan_res.csv", "w", encoding="utf8") as f:
        f.write("card_id,hour,lat,lng,label\n")
        for i, j in res.items():
            label = j["label"]
            poi = j["poi"]
            f.write("{},{},{},{},{}\n".format(poi[0], poi[1], poi[2][0], poi[2][1], label))
