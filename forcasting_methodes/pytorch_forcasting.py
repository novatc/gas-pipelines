from datetime import datetime

import pandas as pd
import torch
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

def pytorch_forcast(df: pd.DataFrame):
    name = df.columns[0]
    df.index = df.iloc[:, 0].apply(to_datetime)

    # turn the index into timestampes
    df["date"] = df.index.map(datetime.timestamp)

    # let x be the co2 and the date_float column
    x = df[["co2", "date"]].values
    y = df["ch4"].values

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
    for epoch in range(1000):
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




