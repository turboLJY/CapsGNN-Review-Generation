import argparse
from texttable import Texttable
from train import trainIters
from evaluate import runTest


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Train the model with corpus")
    parser.add_argument("--load", help="Load the saved model and train")
    parser.add_argument("--test", help="Load the saved model and test")

    parser.add_argument("--aspect_model", help="the saved aspect model")
    parser.add_argument("--review_model", help="the saved review model")

    parser.add_argument("--epochs", type=int, default=400, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--rnn_layers", type=int, default=2, help="Number of layers in encoder and decoder")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size in encoder and decoder")
    parser.add_argument("--embed_size", type=int, default=256, help="embedding size of topic")
    parser.add_argument("--node_size", type=int, default=64, help="embedding size of attribute")
    parser.add_argument("--beam_size", type=int, default=4, help="beam size in decoder")
    parser.add_argument("--gcn_layers", type=int, default=3, help="GCN layers")
    parser.add_argument("--gcn_filters", type=int, default=100, help="GCN layers")

    parser.add_argument("--capsule_size", type=int, default=20, help="capsule size of primary/graph/aspect capsules")
    parser.add_argument("--capsule_num", type=int, default=10, help="number of capsules")

    parser.add_argument("--lr_decay_ratio", type=float, default=0.8, help="learning rate decay ratio")
    parser.add_argument("--lr_decay_epoch", type=int, default=2, help="learning rate decay epoch")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=10 ** -6, help="Weight decay. Default is 10^-6.")

    parser.add_argument("--max_length", type=int, default=50, help="max length of sequence")
    parser.add_argument("--min_length", type=int, default=20, help="min length of sequence")

    parser.add_argument("--save_dir", help="saved directory of model")
    parser.add_argument("--load_file", help="saved model")

    args = parser.parse_args()

    return args


def run(args):

    tab_printer(args)

    learning_rate, lr_decay_epoch, lr_decay_ratio, weight_decay, embed_size, hidden_size, \
        node_size, capsule_size, gcn_layers, gcn_filters, rnn_layers, capsule_num, batch_size, epochs = \
        args.learning_rate, args.lr_decay_epoch, args.lr_decay_ratio, args.weight_decay, args.embed_size, \
        args.hidden_size, args.node_size, args.capsule_size, args.gcn_layers, args.gcn_filters, \
        args.rnn_layers, args.capsule_num, args.batch_size, args.epochs

    if args.train:
        trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, weight_decay, batch_size,
                   rnn_layers, hidden_size, embed_size, node_size, epochs, args.save_dir)

    elif args.load:
        trainIters(args.load, learning_rate, lr_decay_epoch, lr_decay_ratio, weight_decay, batch_size,
                   rnn_layers, hidden_size, embed_size, node_size, epochs, args.save_dir, args.load_file)

    elif args.test:
        runTest(args.test, rnn_layers, hidden_size, embed_size, node_size, capsule_size, gcn_layers, gcn_filters,
                capsule_num, args.aspect_model, args.review_model, args.beam_size,
                args.max_length, args.min_length, args.save_dir)

    else:
        print("mode error!")


if __name__ == "__main__":
    args = parse()
    run(args)
