from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from graph import *
import time
from getmodel import getmodels, getmodel

def parse_args():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    # input files
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output embedding file')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')


    parser.add_argument('--epoch-fac', default=50, type=int,
                        help='epoch-fac * node num in graph = node num per epoch')

    # algorithm parameters
    parser.add_argument('--model-v', required=True,
                        help='The vertex sampling model')

    # APP
    parser.add_argument('--app-jump-factor', default=0.15, type=float,
                        help='Jump factor (APP)')
    parser.add_argument('--app-step', default=80, type=int,
                        help='Maximum number of walking steps(APP)')

    # deepwalk
    parser.add_argument('--degree-power', default=1.0, type=float,
                        help='Bound of degree for sample_v of deepwalk.')
    parser.add_argument('--degree-bound', default=0, type=int,
                        help='Bound of degree for sample_v of deepwalk.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model.')

    # combination
    parser.add_argument('--combine', default=0.5, type=float,
                        help='Combine A and B with how much A.')

    args = parser.parse_args()

    return args

def print_args(args):
    print("==================")
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    print("==================")

def main(args):
    print_args(args)

    print("Reading Graph ...")
    g = Graph()
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)

    model = getmodel(args.model_v, g, args)
    model.gendata(args.output)

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    main(parse_args())
