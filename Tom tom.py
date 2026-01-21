
import requests
import time
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==================================
# 1. API KEY
# ==================================
TOMTOM_API_KEY = "GGhit0Yrj1lDdtymRU6G2aR4of5rtueY"

# ==================================
# 2. LANE COORDINATES
# ==================================
lanes = {
    "North": (18.515146545607323, 73.84229455961288),
    "South": (18.51416364138019, 73.84257162522808),
    "West":  (18.514485380139913, 73.84220282148893)
}

DATA_FILE = "traffic_data.csv"

# print("CSV file location:", os.path.abspath(DATA_FILE))
# C:\Users\mkara\AppData\Local\Programs\Microsoft VS Code\traffic_data.csv


# ==================================
# 3. FETCH TRAFFIC DATA
# ==================================
def get_traffic_data(lat, lon):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "point": f"{lat},{lon}",
        "key": TOMTOM_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        flow = response.json()["flowSegmentData"]
        return flow["currentSpeed"], flow["freeFlowSpeed"]
    else:
        return None, None


# ==================================
# 4. VEHICLE COUNT ESTIMATION
# ==================================
def estimate_vehicle_count(current_speed, free_flow_speed, max_vehicles=40):
    if current_speed is None or free_flow_speed is None:
        return 0

    congestion = (free_flow_speed - current_speed) / free_flow_speed
    vehicles = int(congestion * max_vehicles)

    return max(vehicles, 0)


# ==================================
# 5. SAVE DATA FOR ML
# ==================================
def save_data(lane, vehicle_count, green_time):
    file_exists = os.path.isfile(DATA_FILE)

    data = {
        "Lane": [lane],
        "Vehicle_Count": [vehicle_count],
        "Green_Time": [green_time]
    }

    df = pd.DataFrame(data)

    df.to_csv(DATA_FILE, mode="a", header=not file_exists, index=False)


# ==================================
# 6. TRAIN ML MODEL
# ==================================
def train_ml_model():
    if not os.path.isfile(DATA_FILE):
        return None

    df = pd.read_csv(DATA_FILE)

    if len(df) < 5:
        return None   # not enough data yet

    x = df[["Vehicle_Count"]]
    y = df["Green_Time"]

    model = LinearRegression()
    model.fit(x, y)

    return model


# ==================================
# 7. FALLBACK GREEN TIME (INITIAL)
# ==================================
def fallback_green_time(vehicle_count):
    return int(15 + vehicle_count * 1.2)


# ==================================
# 8. MAIN CONTROLLER
# ==================================
def traffic_signal_controller():
    print("\n Collecting traffic data...\n")

    model = train_ml_model()
    lane_predictions = {}

    for lane, (lat, lon) in lanes.items():
        current_speed, free_flow_speed = get_traffic_data(lat, lon)
        vehicles = estimate_vehicle_count(current_speed, free_flow_speed)

        if model:
            green_time = int(model.predict([[vehicles]])[0])
            ml_used = True
        else:
            green_time = fallback_green_time(vehicles)
            ml_used = False

        save_data(lane, vehicles, green_time)

        lane_predictions[lane] = green_time

        print(f"{lane} Lane:")
        print(f"  Vehicles   : {vehicles}")
        print(f"  Green Time : {green_time} sec")
        print(f"  ML Used    : {ml_used}\n")

    priority_lane = max(lane_predictions, key=lane_predictions.get)

    print(" SIGNAL DECISION ")
    print("Priority Lane :", priority_lane)
    print("Green Time    :", lane_predictions[priority_lane], "seconds")


# ==================================
# 9. AUTOMATION LOOP
# ==================================
if __name__ == "__main__":
    while True:
        traffic_signal_controller()
        print("\n Waiting for next update...\n")
        time.sleep(10)
        