import argparse
from train import trainIters
from evaluate import runTest

def parse():
    parser = argparse.ArgumentParser(description='exp-gen')
    parser.add_argument('-tr', '--train')
    parser.add_argument('-ts', '--test')
    
    parser.add_argument('-an', '--num_contexts', type=int, default=3, help='how many contexts? (e.g., user, product, rating)')
    parser.add_argument('-es', '--embed_size', type=int, default=512, help='embedding size of topic')
    parser.add_argument('-as', '--attr_size', type=int, default=512, help='embedding size of attribute, e.g. user, product')
    
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-ly', '--layer', type=int, default=2, help='Number of layers in encoder and decoder')
    parser.add_argument('-hs', '--hidden_size', type=int, default=512, help='Hidden size in encoder and decoder')
    parser.add_argument('-bm', '--beam_size', type=int, default=4, help='beam size in decoder')
    parser.add_argument('-or', '--overall', type=int, default=5, help='overall scale, change with the dataset')
    parser.add_argument('-dr', '--lr_decay_ratio', type=float, default=0.8, help='learning rate decay ratio')
    parser.add_argument('-de', '--lr_decay_epoch', type=int, default=5, help='learning rate decay epoch')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00002, help='Learning rate')

    parser.add_argument('-mx', '--max_length', type=int, default=10, help='max length of sequence')
    parser.add_argument('-mn', '--min_length', type=int, default=2, help='min length of sequence')

    parser.add_argument('-sd', '--save_dir', help='Model directory')
    parser.add_argument('-ld', '--load', help='Start with loaded model')
    parser.add_argument('-md', '--model', help='Saved model name')  

    args = parser.parse_args()
    return args

def get_model_info(filename):
    layers, hidden, batch_size = filename.split('/')[-2].split('_')
    return int(n_layers), int(hidden_size)

def run(args):
    learning_rate, lr_decay_epoch, lr_decay_ratio, n_layers, hidden_size, embed_size, \
        attr_size, attr_num, batch_size, beam_size, overall, max_length, min_length, save_dir = \
            args.learning_rate, args.lr_decay_epoch, args.lr_decay_ratio, args.layer, args.hidden_size, args.embed_size, \
                args.attr_size, args.num_contexts, args.batch_size, args.beam_size, args.overall, args.max_length, args.min_length, args.save_dir
        
    if args.train and not args.load:
        trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, \
                n_layers, hidden_size, embed_size, attr_size, attr_num, overall, save_dir)
    elif args.load:
        n_layers, hidden_size = get_model_info(args.load)
        trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, \
                n_layers, hidden_size, embed_size, attr_size, attr_num, overall, save_dir, loadFilename=args.load)
    elif args.test: 
        n_layers, hidden_size = get_model_info(args.model)
        runTest(args.test, n_layers, hidden_size, embed_size, attr_size, attr_num, overall, \
            args.model, beam_size, max_length, min_length, save_dir)

if __name__ == '__main__':
    args = parse()
    run(args)
