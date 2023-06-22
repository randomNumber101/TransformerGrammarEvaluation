import os
import typing
import torch

dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, ".." + os.sep + "data" + os.sep)
result_folder = os.path.join(dirname, ".." + os.sep + ".." + os.sep + "server-import" + os.sep)


class HyperParams:
    def __init__(self):
        # default values
        self.device = "cpu"
        self.batch_size = 32
        self.input_size = -1
        self.num_epochs = 5

        self.learning_rate = 1e-5
        self.max_norm = 1.0  # Used for gradient clipping

        # Printing
        self.print_every = 5


def overwrite_params(hps: HyperParams, overwrite: typing.Dict[str, object]):
    for (k, val) in overwrite.items():
        if val is None:
            continue
        if k == "device":
            hps.device = str(val)
        elif k == "batch_size":
            hps.batch_size = int(val)
        elif k == "input_size":
            hps.input_size = int(val)
        elif k == "num_epochs":
            hps.num_epochs = int(val)
        elif k == "learning_rate":
            hps.learning_rate = float(val)
        elif k == "max_norm":
            hps.max_norm = float(val)
        elif k == "print_every":
            hps.print_every = int(val)
    return hps

def save(name, model, optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(result_folder, name + ".pth"))
    print("Saved model and optimizer to " + result_folder + " with name: " + name + ".")
