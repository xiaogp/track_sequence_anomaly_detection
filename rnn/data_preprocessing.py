# -*- coding: utf-8 -*-


import time
import pickle
from utils import load_yaml_config

file_path = "./todData/sample_data.csv"


def data_preprocessing():
    trackDict = {}

    with open(file_path, "r", encoding="utf8") as f:
        next(f)
        for line in f.readlines():
            camera_id, alarm_time = [int(x) for x in line.strip().split(",")[1:3]]
            card_id = line.strip().split(",")[6]
            timestamp = float(alarm_time / 1000)
            day = time.localtime(timestamp).tm_mday
            hour = time.localtime(timestamp).tm_hour
            minute = time.localtime(timestamp).tm_min
            tim = round(hour + minute / 60, 2)

            # {card: {day: [(time, location))]]}
            trackDict.setdefault(card_id, {})
            trackDict[card_id].setdefault(day, [])
            trackDict[card_id][day].append((tim, camera_id))

    # list根据时间从小到大排序,{card: {day: [(sorted(time), location))]]}
    sortedTrackDict = {k: {kk: sorted(vv) for kk, vv in v.items()} for k, v in trackDict.items()}
    pickle.dump(sortedTrackDict, open(pickle_path, "wb"))

    # 轨迹序列保存为txt文件，hour_location,hour_location,hour_location...
    with open(output_path, "w", encoding="utf8") as f:
        for card, dayDict in sortedTrackDict.items():
            for day, hourList in dayDict.items():
                for hour, loc in hourList:
                    f.write(str(round(hour)) + "_" + str(loc) + " ")
                f.write("\n")


if __name__ == "__main__":
    config = load_yaml_config("config.yml")
    file_path = config["data"]["file_path"]
    output_path = config["data"]["output_path"]
    pickle_path = config["data"]["pickle_path"]
    data_preprocessing()
