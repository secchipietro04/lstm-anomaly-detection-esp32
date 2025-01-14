import threading
import queue
import serial
import matplotlib.pyplot as plt
import time
import re

# Serial port configuration
SERIAL_PORT = "/dev/ttyACM1"  # Adjust based on your setup
BAUD_RATE = 115200

# Shared queues for data processing
data_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent overflow
graph_queues = {0: queue.Queue(maxsize=100), 1: queue.Queue(maxsize=100), 2: queue.Queue(maxsize=100)}
logging_queue = queue.Queue(maxsize=100)  # Queue for logging data
processed_data = {0: {"time": [], "x": [], "y": [], "z": []},
                  1: {"time": [], "x": [], "y": [], "z": []},
                  2: {"time": [], "x": [], "y": [], "z": []}}

# Plotting configuration
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
graphs = {
    0: {"ax": axs[0], "label": "Gyro"},
    1: {"ax": axs[1], "label": "Acceleration"},
    2: {"ax": axs[2], "label": "Linear Acceleration"},
}

lines = {}
for graph_id, graph_info in graphs.items():
    ax = graph_info["ax"]
    ax.set_title(graph_info["label"])
    ax.set_xlim(0, 10)
    ax.set_ylim(-5000, 8000)
    lines[graph_id] = {
        "x": ax.plot([], [], label="X")[0],
        "y": ax.plot([], [], label="Y")[0],
        "z": ax.plot([], [], label="Z")[0],
    }
    ax.legend()

# Thread 1: Data Reader
def data_reader():
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    try:
        while True:
            line = ser.readline().decode('utf-8').split("ISM330BX: ")[-1].strip()
            line = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', line)
            data_queue.put(line)
    except serial.SerialException as e:
        print("Serial error:", e)
    except KeyboardInterrupt:
        ser.close()

# Thread 2: Data Distributor
def data_distributor():
    while True:
        line = data_queue.get()
        try:
            graph_id = int(line.split(',')[0])  # Extract graph ID
            if graph_id in graph_queues:
                graph_queues[graph_id].put_nowait(line)
        except (ValueError, IndexError, queue.Full):
            print("Invalid or missed data:", line)

# Threads 3-5: Data Processors
def data_processor(graph_id):
    while True:
        line = graph_queues[graph_id].get()
        try:
            values = list(map(int, line.split(',')))
            _, gyro_x, gyro_y, gyro_z = values

            # Append data
            current_time = time.time()
            data = processed_data[graph_id]
            data["time"].append(current_time)
            data["x"].append(gyro_x)
            data["y"].append(gyro_y)
            data["z"].append(gyro_z)

            # Put data into the logging queue for logging
            logging_queue.put((graph_id, current_time, gyro_x, gyro_y, gyro_z))

            # Keep last 10 seconds of data
            while current_time - data["time"][0] > 10:
                data["time"].pop(0)
                data["x"].pop(0)
                data["y"].pop(0)
                data["z"].pop(0)
        except ValueError:
            print(f"Invalid data for graph {graph_id}: {line}")

# Thread 6: Data Logger
def data_logger():
    with open("/dev/null", "a") as log_file:
        while True:
            data = logging_queue.get()
            if data is None:  # Exit signal
                break
            graph_id, current_time, gyro_x, gyro_y, gyro_z = data
            log_file.write(f"{graph_id},{current_time},{gyro_x},{gyro_y},{gyro_z}\n")
            log_file.flush()  # Ensure data is written immediately

# Main thread: Plot Updater
def update_plot():
    while True:
        for graph_id, graph_info in graphs.items():
            ax = graph_info["ax"]
            data = processed_data[graph_id]

            if data["time"]:
                lines[graph_id]["x"].set_data(data["time"], data["x"])
                lines[graph_id]["y"].set_data(data["time"], data["y"])
                lines[graph_id]["z"].set_data(data["time"], data["z"])

                # Update axes limits
                ax.set_xlim(data["time"][0], data["time"][-1])
                y_min = min(data["x"] + data["y"] + data["z"]) - 1000
                y_max = max(data["x"] + data["y"] + data["z"]) + 1000
                ax.set_ylim(y_min, y_max)

        plt.pause(0.000001)
        time.sleep(1)

# Start threads
threads = [
    threading.Thread(target=data_reader, daemon=True),
    threading.Thread(target=data_distributor, daemon=True),
]

# Add processing threads
for graph_id in range(3):
    threads.append(threading.Thread(target=data_processor, args=(graph_id,), daemon=True))

# Add logging thread
threads.append(threading.Thread(target=data_logger, daemon=True))

# Start all threads
for thread in threads:
    thread.start()

# Start plotting in the main thread
try:
    update_plot()
except KeyboardInterrupt:
    print("Shutting down...")
finally:
    # Signal the logger thread to exit
    logging_queue.put(None)
