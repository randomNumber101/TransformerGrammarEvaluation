import os
class HyperParams:
    def __init__(self):
        self.device = "cpu"
        self.batch_size = 32
        self.input_size = 128
        self.num_epochs = 5

        self.learning_rate = 1e-5
        self.max_norm = 1.0  # Used for gradient clipping

        # Printing
        self.print_every = 5


dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, ".." + os.sep + "data" + os.sep)
result_folder = os.path.join(dirname, ".." + os.sep + ".." + os.sep + "server-import" + os.sep)


