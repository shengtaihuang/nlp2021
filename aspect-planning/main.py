import argparse
import train, evaluate

def get_arguments():
    parser = argparse.ArgumentParser(description='exp-gen')
    parser.add_argument('-tr', '--train')
    parser.add_argument('-ts', '--test')
    
    parser.add_argument('-c', '--num_contexts', type=int, default=3, help='how many contexts? (e.g., user, product, rating)')
    parser.add_argument('-aes', '--embed_size', type=int, default=512, help='embedding size of aspects')
    parser.add_argument('-ces', '--context_embed_size', type=int, default=512, help='embedding size of contexts, e.g. user, product')
    
    parser.add_argument('-bs', '--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-l', '--layer', type=int, default=2, help='how many layers?')
    parser.add_argument('-hd', '--hidden_size', type=int, default=512, help='hidden size?')
    parser.add_argument('-or', '--overall', type=int, default=5, help='overall scale, change with the dataset')
    parser.add_argument('-d', '--lr_decay_ratio', type=float, default=0.9, help='learning rate')
    parser.add_argument('-de', '--lr_decay_epoch', type=int, default=5, help='learning rate and decay epoch')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    parser.add_argument('-b', '--beam_size', type=int, default=4, help='beam size in the decoder')

    parser.add_argument('-mx', '--max_length', type=int, default=10, help='max length of sequence')
    parser.add_argument('-mn', '--min_length', type=int, default=2, help='min length of sequence')

    parser.add_argument('-sd', '--save_dir', help='Model directory')
    parser.add_argument('-ld', '--load', help='Start with loaded model')
    parser.add_argument('-md', '--model', help='Saved model name')  

    args = parser.parse_args()
    return args

def get_model_info(filename):
    n_layers, hidden_size, batch_size = filename.split('/')[-2].split('_')
    print('layers, hidden, batch_size: ', n_layers, hidden_size, batch_size)
    return int(n_layers), int(hidden_size)

def run(args):
    learning_rate, lr_decay_epoch, lr_decay_ratio, n_layers, hidden_size, embed_size, \
        context_embed_size, attr_num, batch_size, beam_size, overall, max_length, min_length, save_dir = \
            args.learning_rate, args.lr_decay_epoch, args.lr_decay_ratio, args.layer, args.hidden_size, args.embed_size, \
                args.context_embed_size, args.num_contexts, args.batch_size, args.beam_size, args.overall, args.max_length, args.min_length, args.save_dir
        
    if args.train and not args.load:
        train.trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, \
                n_layers, hidden_size, embed_size, context_embed_size, attr_num, overall, save_dir)
    elif args.load:
        n_layers, hidden_size = get_model_info(args.load)
        train.trainIters(args.train, learning_rate, lr_decay_epoch, lr_decay_ratio, batch_size, \
                n_layers, hidden_size, embed_size, context_embed_size, attr_num, overall, save_dir, loadFilename=args.load)
    elif args.test: 
        n_layers, hidden_size = get_model_info(args.model)
        evaluate.runTest(args.test, n_layers, hidden_size, embed_size, context_embed_size, attr_num, overall, \
            args.model, beam_size, max_length, min_length, save_dir)

if __name__ == '__main__':
    args = get_arguments()
    run(args)
