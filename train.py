from argparse import ArgumentParser
import torch
import os
import model
import random


def evaluate(model, config):
    model.eval()
    time_test = data_process('time-test.txt')
    if config.generate_type:
        type_test = generate_type(time_test)
    else:
        type_test = data_process("event_test.txt")
    loss_total = 0
    loss_type = 0
    loss_time = 0
    for index in range(len(time_test)):
            batch = (torch.tensor([time_test[index]], dtype=torch.float32), torch.tensor([type_test[index]]))
            loss, loss1, loss2 = model.train_batch(batch, device)
            loss_total += loss
            loss_type += loss1
            loss_time += loss2
    loss_total /= len(time_test)
    loss_type /= len(time_test)
    loss_time /= len(time_test)    
    print("During evaluation, the losses are {}, {}, {}".format(loss_total, loss_type, loss_time))

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=1)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--opt", type=str, default='Adam')
    parser.add_argument("--generate_type",
                        help="For generated experiment, use true for generate_type",
                        type=bool, default=True)
    parser.add_argument("--read_model", type=bool, default=False)
    config = parser.parse_args()

    time_train = data_process('time-train.txt')
    if config.generate_type:
        type_train = generate_type(time_train)
    else:
        type_train = data_process("event_train.txt")
    print("training file processed.")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of CUDAs(cores): ", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())

    if config.read_model:
        model = torch.load("model.pt")
    else:
        model = model.RMTPP(config, device)

    for epc in range(config.epochs):
        c = list(zip(time_train, type_train))
        random.shuffle(c)
        time_train, type_train = zip(*c)
        loss_total = 0
        loss_type = 0
        loss_time = 0
        for index in range(len(time_train)):
            batch = (torch.tensor([time_train[index]], dtype=torch.float32), torch.tensor([type_train[index]]))
            loss, loss1, loss2 = model.train_batch(batch, device)
            loss_total += loss
            loss_type += loss1
            loss_time += loss2
        print("In epochs {0}, total loss: {1}, type loss: {2}, time loss: {3}".format(
            epc, loss_total/len(time_train), loss_type/len(time_train), loss_time/len(time_train)
        ))
        print("saving model")
        torch.save(model, "model.pt")
        evaluate(model, config)
    print("training done!")