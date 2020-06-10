import torch
import torch.nn as nn


class RMTPP(nn.Module):
    def __init__(self, config, device):
        super(RMTPP, self).__init__()
        self.config = config
        self.n_class = config.n_class
        self.hid_dim = config.hid_dim
        self.embed = nn.Linear(in_features=self.n_class, out_features=1)
        self.rnn = nn.RNN(input_size=2, hidden_size=config.hid_dim,
                          batch_first=True, num_layers=config.n_layers,bidirectional=False, nonlinearity='relu')
        
        self.event_linear = nn.Linear(in_features=config.hid_dim, out_features=self.n_class, bias=True)
        self.time_linear = nn.Linear(in_features=config.hid_dim, out_features=1, bias=True)

        self.event_criterion = nn.CrossEntropyLoss()
        self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.time_criterion = self.RMTPPLoss
        self.optimizer = self.set_optimizer(config.opt, config.lr, config.momentum)

    def RMTPPLoss(self, hidden_things, time_duration):
        loss = torch.mean(hidden_things+self.intensity_w * time_duration + self.intensity_b +
                         (torch.exp(hidden_things + self.intensity_b) -
                          torch.exp(hidden_things+self.intensity_w * time_duration + self.intensity_b))/self.intensity_w)
        return -loss

    def forward(self, time_input, event_input):
        event_input = torch.nn.functional.one_hot(event_input, num_classes=self.n_class)
        event_input = event_input.float()
        event_embedding = self.embed(event_input)
        rnn_input = torch.cat((event_embedding, time_input.unsqueeze(-1)), dim=-1)
        hidden_state, _ = self.rnn(rnn_input)
        #print(hidden_state)
        event_out = self.event_linear(hidden_state)
        time_out = self.time_linear(hidden_state)
        return event_out, time_out

    def set_optimizer(self, opt, lr, momentum):
        if opt == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            return torch.optim.Adam(self.parameters(), lr=lr)

    def train(self, batch, device):
        time_tensor, event_tensor = batch
        time_tensor.to(device)
        event_tensor.to(device)
        time_input, time_duration = time_tensor[:, :-1], time_tensor[:, -1]
        event_input, event_target = event_tensor[:, :-1], event_tensor[:, -1]
        event_out, time_out = self.forward(time_input, event_input)
        event_out = event_out[:,-1:].reshape(-1, self.n_class)
        loss1 = self.event_criterion(event_out, event_target)
        time_out = time_out[:,-1]
        loss2 = self.time_criterion(time_out.reshape(-1,1), time_duration.reshape(-1,1))
        loss =  loss1 + loss2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), loss1.item(), loss2.item()

    def predict(self, batch, device):
        time_tensor, event_tensor = batch
        time_tensor.to(device)
        event_tensor.to(device)
        time_input, time_duration = time_tensor[:,:-1], time_tensor[:,-1]
        event_input, event_target = event_tensor[:,:-1], event_tensor[:,-1]
        event_out, time_out = self.forward(time_input, event_input)
        time_out = time_out[:,-1]
        event_pred = nn.functional.softmax(event_out[:,-1])
        event_pred = torch.max(event_pred, dim=-1)[1].tolist()
        return event_pred, time_out
