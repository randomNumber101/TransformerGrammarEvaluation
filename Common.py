import argparse
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


def loadParams():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--set_name', type=str, help="The data set", required=True)
    parser.add_argument('-m', '--model', type=str, help="Model to use [BART, LSTM]", default="BART"),
    parser.add_argument('-bs', '--batch_size', type=int, help="Batch size")
    parser.add_argument('-is', '--input_size', type=int, help="Input (and output) Length of the network")
    parser.add_argument('-epochs', '--num_epochs', type=int, help="Epoch count")
    parser.add_argument('-lr', '--learning_rate', type=float, help="Learning rate")
    parser.add_argument('-mn', "--max_norm", type=float, help="Maximal gradient for gradient clipping")
    parser.add_argument('-pe', '--print_every', type=int, help="Frequency of logging results")
    parser.add_argument('-test', '--test_mode', help="Test mode. Deactivated wandb.", action='store_true')
    parser.add_argument('-l', '--layers', type=int, help="Number of layers", default=-1)


    args = parser.parse_args()

    # Set script Hyper Parameters
    hp = HyperParams()
    overwrite_params(hp, vars(args))
    hp.device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if CUDA is available

    print(f"hyper parameters: {vars(hp)}")
    return args, hp



def save(name, model, optimizer):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(result_folder, name + ".pth"))
    print("Saved model and optimizer to " + result_folder + " with name: " + name + ".")
