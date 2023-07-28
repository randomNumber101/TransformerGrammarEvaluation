import argparse

# [TASK] [-d SET] [-bs BS] [-is INPUT_SIZE] [-l layers][-EPOCHS] [-model MODELS] [-tok TOKENIZATION]


parser = argparse.ArgumentParser()
parser.add_argument('-app', '--append', nargs='+', type=str, help="Further static string to add to script calls")
parser.add_argument('-script', '--script', type=str, help="The data set", required=True)
parser.add_argument('-sets', '--sets', nargs='+', type=str, help="The data set", required=True)
parser.add_argument('-m', '--models', nargs='+', type=str, help="Model to use [TRANSF, LSTM]", default="TRANSF"),
parser.add_argument('-bs', '--batch_sizes', nargs='+', type=int, help="Batch size")
parser.add_argument('-is', '--in_sizes', nargs='+', type=int, help="Input (and output) Length of the network")
parser.add_argument('-l', '--layer_counts', nargs='+', type=int, help="Number of layers")
parser.add_argument('-epochs', '--epochs', nargs='+', type=int, help="Epoch counts")
parser.add_argument('-tok', '--tokenizations', nargs='+', type=str,
                    help="Tokenization strategies [bpe, words, words_bpe]",
                    default="bpe")
parser.add_argument('-lr', '--learning_rate', type=float, help="Learning rate")
parser.add_argument('-mn', "--max_norm", type=float, help="Maximal gradient for gradient clipping")
parser.add_argument('-pe', '--print_every', type=int, help="Frequency of logging results")
parser.add_argument('-test', '--test', help="Test mode. Deactivates wandb.", action='store_true')
args = parser.parse_args()

script = args.script
sets = args.sets
epochs = args.epochs
models = args.models
tokenizations = args.tokenizations

configurable = ["sets", "batch_sizes", "in_sizes", "layer_counts", "epochs", "models", "tokenizations", "test"]

for key in configurable:
    value = getattr(args, key)
    if not isinstance(value, list):
        setattr(args, key, [value])


def create_single_call(script, train_set=None, batch_size=None, in_size=None, layer_count=None, epochs=None, model=None,
                       tokenization=None):
    train_set = f"-d {train_set} " if train_set is not None else ""
    batch_size = f"-bs {batch_size} " if batch_size is not None else ""
    in_size = f"-is {in_size} " if in_size is not None else ""
    layer_count = f"-l {layer_count} " if layer_count is not None else ""
    epochs = f"-epochs {epochs} " if epochs is not None else ""
    model = f"-m {model} " if model is not None else ""
    tokenization = f"-tok {tokenization} " if tokenization is not None else ""
    test = "-test " if args.test[0] is not None and args.test[0] else ""

    call = f"{script} {train_set}{batch_size}{in_size}{layer_count}{epochs}{model}{tokenization}{test}"
    call += " && \\ \nrm -r models/wandb && \\\n"
    return call


output = ""
for train_set in args.sets:
    for batch_size in args.batch_sizes:
        for in_size in args.in_sizes:
            for layer_count in args.layer_counts:
                for epoch in args.epochs:
                    for model in args.models:
                        for tokenization in args.tokenizations:
                            output += create_single_call(args.script, train_set, batch_size, in_size, layer_count,
                                                         epoch, model, tokenization)

print(output)
