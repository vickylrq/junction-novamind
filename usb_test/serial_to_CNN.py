import serial
from serial.tools import list_ports
import numpy as np
import re
import csv
import pandas as pd

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
    t = serialList()
    ser = serial.Serial("COM9", 9600, timeout=0.01)
    data_set = []
    data_from_Serial = ser.readline()
    if data_from_Serial:
        rec_str = data_from_Serial.decode('utf-8')
        data = extract_numbers(rec_str)
        if len(data) != 9:
            continue
        data_set.append(data)
        if len(data_set) > 100:
            break


if __name__=='__main__':
    while True:
        data_out = serial_to_CNN()









