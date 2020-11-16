import configargparse
import pdb

def pair(arg):
    return [float(x) for x in arg.split(',')]

def get_args():
    parser = configargparse.ArgParser(default_config_files=[])
    parser.add("--flip_rate", type=float, default="0.5", help="Horizontal random flip rate, default=0.5")
    parser.add("--brightness", type=float, default="0.1", help="Brightness adjustion, default=0.1")
    parser.add("--rd_crop", type=int, default=(50,100), help="Random crop size, default=(50,100)") 
    parser.add("--batch_size", type=int, default=4, help="Batch_sizd=e, default=4") 
    parser.add("--lr", type=float, default=0.1, help="Learning rate, default=0.1")
    parser.add("--momentum", type=float, default=0.9, help="Momentum, default=0.9")
    parser.add("--weight_decay", type=float, default=0.0002, help="L2 norm regularization strength, default=0.0002")
    parser.add("--decay_bd", type=int, default=[50,75], help="Learning rate decay boundary, default=[50,75]")
    parser.add("--gamma", type=float, default=0.1, help="Learning rate decay rate, default=0.1")
    parser.add("--num_epochs", type=int, default=1, help="Training epochs, default=100")

    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    print(get_args())
    pdb.set_trace()

# TODO Default for model_dir
# TODO Need to update the helps
