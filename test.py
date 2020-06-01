from argparse import ArgumentParser
import torch
import os
import numpy as np
from scipy import integrate
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

def data_process(file_name):
    f = open(file_name,'r')
    time_data = []
    file_data = f.readlines()
    f.close()
    for line in file_data:
        data = line.split(" ")
        a_list = []
        for i in range(len(data)):
            if data[i] != "\n":
                a_list.append(float(data[i]))
        time_data.append(a_list)
    return time_data


def generate_type(time_data):
    type_data = []
    for line in time_data:
        new_line = []
        for item in line:
            new_line.append(1)
        type_data.append(new_line)
    return type_data


def equation(time_var, time_cif, start_time, w, b):
    time_guess = time_var*np.exp(time_cif+w*(time_var-start_time)+b+
                                 (np.exp(time_cif+b)-np.exp(time_cif+w*(time_var-start_time)+b))/w)
    return time_guess


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generate_type",
                        help="For generated experiment, use true for generate_type",
                        type=bool, default=True)
    parser.add_argument("--pre_test_numb",
                        help="Number of pre_test data to be put into the model.",
                        type=int, default=55)
    parser.add_argument("--length_to_predict",
                        type=int, default=100)
    config = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of CUDAs(cores): ", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())
    time_test = data_process('time-test.txt')
    if config.generate_type:
        type_test = generate_type(time_test)
    else:
        type_test = data_process("event_test")
    print("testing file processed.")
    time_test = time_test[0]
    type_test = type_test[0]

    model = torch.load("model.pt")

    number = config.pre_test_numb + 1
    time = time_test[:number]
    types = type_test[:number]
    for i in range(config.length_to_predict):
        batch = (torch.tensor([time], dtype=torch.float32), torch.tensor([types]))
        event_pred, time_cif = model.predict(batch, device)
        time_cif = time_cif.item()
        intensity_w = model.intensity_w.item()
        intensity_b = model.intensity_b.item()
        func = lambda x: equation(x, time_cif, time[-1], intensity_w, intensity_b)
        time_pred = integrate.quad(func,time[-1], np.inf)
        time.append(time_pred[0])
        types.append(event_pred)
    print("predict_time: ",time[number:])
    print("actual_time: ", time_test[number:])
    print("predict_type: ", types[number:])
    print("actual_type: ", types[number:])

    f = open("predict-time.txt", "w")
    for t in time[number:]:
      f.write(str(t))
      f.write(" ")
    f.close()
    f = open("predict_type.txt", "w")
    for item in types[number:]:
      f.write(str(item))
      f.write(" ")
    f.close()

    print("calculating RMSE: ")
    print(sqrt(mean_squared_error(time[number:], time_test[number:])))
    print("generating_time_interval_plot:")

    difference = []
    for j in range(number, len(time)):
        difference.append(time[j]-time[j-1])
    plt.plot(range(100),difference)
    plt.ylim(top=0.22, bottom=0)
    plt.savefig("time_duration.png")
    print(difference)


    