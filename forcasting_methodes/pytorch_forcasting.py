from datetime import datetime

import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn

# create the model
class TimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 12)
        self.fc2 = nn.Linear(12, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def to_datetime(date_string):
    date_format = "%b %Y"
    return datetime.strptime(date_string, date_format)

def pytorch_forcast(dfdf: pd.DataFrame):
    df_copy = dfdf.copy()
    df_copy.reset_index(inplace=True)
    name = df_copy.columns[0]
    df_copy.index = df_copy.iloc[:, 0].apply(to_datetime)

    # turn the index into timestampes
    df_copy["date"] = df_copy.index.map(datetime.timestamp)

    # let x be the co2 and the date_float column
    x = df_copy[["co2", "date"]].values
    y = df_copy["ch4"].values
    date = df_copy["date"].values

    # convert the data to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    print(x_train.shape, y_train.shape)

    model = TimeSeriesModel()

    # define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # train the model for 100 epochs
    for epoch in range(100):
        # forward pass
        y_pred = model(x_train)

        # compute the loss
        loss = criterion(y_pred, y_train)

        # zero the gradients
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

    # make predictions on the test data
    predictions = model(x_test)
    performance = nn.MSELoss()(predictions, y_test)
    print("PyTroch Model loss:", performance.item())
    time = date[-len(predictions):]

    plot_forcast(predictions.detach().numpy(), y_test, time, name)

def plot_forcast(predictions, accurate_values, time, name):
    time = [datetime.fromtimestamp(x) for x in time]

    fig, ax = plt.subplots()
    ax.scatter(time, predictions, label="predictions", color="red")
    ax.scatter(time, accurate_values, label="actual", color="blue")

    plt.xlabel("Time")
    plt.ylabel("CH4 emissions")
    plt.title("Predicted output over time")

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    plt.legend()
    plt.savefig('clean/images/pytorch/' + name + '.png')
    plt.clf()



