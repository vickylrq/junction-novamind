import pandas as pd

def read(file_path):
    data = pd.read_csv(file_path,encoding='utf-8',skiprows=4,usecols=[4,5,6,7,8,9,10,11,12])
    # print(data.values[1:,])
    acc = data.values[1:,0:3]
    ang_rate =data.values[1:,3:6]
    mag_field = data.values[1:,6:9]
    
    return acc,ang_rate,mag_field

if __name__ == "__main__":
    file_path = "Data_Log_2023_11_10_19_49_05.csv"
    a,b,c = read(file_path)
    print(b)