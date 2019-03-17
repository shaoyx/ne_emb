from __future__ import print_function
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from graph import *

import pickle

def create_link_dataset(args):
    """
    Create and cache train & test graphs.
    Will load from cache if exists unless --regen option is given.

    :param args:
    :return:
        Gtrain, Gtest: Train & test graphs
    """
    # Remove half the edges, and the same number of "negative" edges

    # Create random training and test graphs with different random edge selections

    print("Generating link prediction graphs")
    # Train graph embeddings on graph with random links
    Gtrain = Graph(prop_pos=args.prop_pos,
                   prop_neg=args.prop_neg,
                   prop_neg_tot=args.prop_neg_tot)
    if args.graph_format == 'adjlist':
        Gtrain.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        Gtrain.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)
    Gtrain.generate_pos_neg_links()

    cache_data = {'g_train': Gtrain}
    with open(args.output, 'wb') as f:
        pickle.dump(cache_data, f)
    with open("{}.{}".format(args.output, "edgelist"), "w") as f:
        edges = Gtrain.G.edges()
        for edge in edges:
            f.write("{} {}\n".format(edge[0], edge[1]))

def parse_args():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output data set file')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')

    parser.add_argument('--prop-pos', default=0.5, type=float,
                        help='proportion of positive edges for link prediction')
    parser.add_argument('--prop-neg', default=0.5, type=float,
                        help='proportion of negative edges for link prediction')
    parser.add_argument('--prop-neg-tot', default=1.0, type=float,
                        help='total proportion of negative edges for link prediction')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    create_link_dataset(args)