import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports
import re


def extract_numbers(input_string):
    # Use regular expression to find all occurrences of patterns like '=number'
    numbers = re.findall(r'=-?[\d]+', input_string)

    # Remove the '=' symbol and convert to integers
    numbers = [int(num.replace('=', '')) for num in numbers]

    return np.array(numbers)


def serialList():
    # Scan and return a list of port numbers
    comList = []
    portList = list(list_ports.comports())
    portList.sort(reverse=True)
    for port in portList:
        port_str = '%s:%s' % (port[0], port[1])
        comList.append(port_str)
    return comList


if __name__=='__main__':
    t = serialList()
    print(t)
    ser = serial.Serial("COM9", 9600, timeout=0.01)
    data_set = []

    while True:
        data_from_Serial = ser.readline()
        if data_from_Serial:
            rec_str = data_from_Serial.decode('utf-8')
            data = extract_numbers(rec_str)
            if len(data) != 9:
                continue
            data_set.append(data)
            print(len(data_set))
            if len(data_set) > 2000:
                break

    # Convert the list of numpy arrays to a pandas DataFrame
    df = pd.DataFrame(data_set, columns=['ACC_X', 'ACC_Y', 'ACC_Z', 'GYR_X', 'GYR_Y', 'GYR_Z', 'MAG_X', 'MAG_Y', 'MAG_Z'])

    # Save to CSV file
    csv_file_path = 'still.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Data saved to {csv_file_path}")

    # not moving:0
    # jump:1
    # rotate:2
    # circle:3
    # front_back:4