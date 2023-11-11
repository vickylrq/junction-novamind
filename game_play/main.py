import serial
from serial.tools import list_ports
import numpy as np
import re
import pandas as pd
import csv
import torch
import pyautogui
import time


from model import SimpleCNN2D

def extract_numbers(input_string):
    # Use regular expression to find all occurrences of patterns like '=number'
    numbers = re.findall(r'=-?[\d]+', input_string)

    # Remove the '=' symbol and convert to integers
    numbers = [int(num.replace('=', '')) for num in numbers]

    return np.array(numbers)


def serialList():  # 扫描获取端口号列表，并以列表形式返回
    comList = []
    portList = list(list_ports.comports())  # 获取当前可用串口信息
    portList.sort(reverse=True)
    for port in portList:
        # comList.append(port[0])
        port_str = port[1]  # 将串口信息格式重新调整，便于显示
        port_str = '%s:%s' % (port[0], port_str)
        comList.append(port_str)  # 将调整后的串口信息加入列表中
    return comList


def serial_to_CNN():
    # t = serialList()  # show ports
    ser = serial.Serial("COM9", 9600, timeout=0.01)
    data_set = []
    while True:
        data_from_Serial = ser.readline()
        if data_from_Serial:
            rec_str = data_from_Serial.decode('utf-8')
            data = extract_numbers(rec_str)

            # eliminate invalid data
            if len(data) != 9:
                continue

            # Send 10 samples to CNN
            data_set.append(data)
            # print(data)
            if len(data_set) == 10:
                break
    return data_set


if __name__ == '__main__':
#     t = serialList()
#     ser = serial.Serial("COM9", 9600, timeout=0.01)
#     data_set= []
#     while True:
#         data_from_Serial = ser.read_all()
#         if data_from_Serial:
#             rec_str = data_from_Serial.decode('utf-8')
#             data = extract_numbers(rec_str)
#
#             # eliminate invalid data
#             if len(data) != 9:
#                 continue
#
#             # Send 10 samples to CNN
#             data_set.append(data)
#             # print(data)
#             if len(data_set) == 10:
#                 data_to_CNN = data_set
#                 data_set = []
#                 break

    model = SimpleCNN2D(6)
    model.load_state_dict(torch.load('cnn_model_6.pth'))
    time.sleep(5)
    while True:
        data = np.array(serial_to_CNN())
        # print(data)
        data = torch.from_numpy(data[np.newaxis, :]).float()
        pred = torch.argmax(model(data)).numpy()

        if pred == 0 or pred == 5:      # STAY STILL:
            print("Stay still!")
            continue
        elif pred == 1:                 # LEFT:
            print("GO Left!")
            pyautogui.keyDown('a')
            time.sleep(1)
            pyautogui.keyUp('a')
        elif pred == 2:                 # RIGHT:
            print("GO Right!")
            pyautogui.keyDown('d')
            time.sleep(1)
            pyautogui.keyUp('d')
        elif pred == 3:                 # JUMP:
            print("Jump!")
            pyautogui.keyDown('space')
            time.sleep(1)
            pyautogui.keyUp('space')
        else:                           # FRONT AND BACK:
            print("Attack!")
            pyautogui.keyDown('w')
            time.sleep(1)
            pyautogui.keyUp('w')
            # time.sleep(0.01)
            # pyautogui.keyUp('left')

    # # For test
    # data = serial_to_CNN()
    # # Convert the list of numpy arrays to a pandas DataFrame
    # df = pd.DataFrame(data, columns=['ACC_X', 'ACC_Y', 'ACC_Z', 'GYR_X', 'GYR_Y', 'GYR_Z', 'MAG_X', 'MAG_Y', 'MAG_Z'])
    #
    # # Save to CSV file
    # csv_file_path = 'data.csv'
    # df.to_csv(csv_file_path, index=False)
    # print(f"Data saved to {csv_file_path}")