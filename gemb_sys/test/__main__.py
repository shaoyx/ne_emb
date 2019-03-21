from __future__ import print_function
import random
from collections import defaultdict

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from classify import Classifier, read_node_label\

from link import *
from reconstr import reconstr
from clustering import clustering, modularity

# parse_args is moved to getmodel.py
def parse_args():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    # input files
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    # parser.add_argument('--output',
    #                     help='Output representation file')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    # parser.add_argument('--feature-file', default='',
    #                    help='The file of node features')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--embedding-file', required=True,
                        help='Pretrained embedding file')

    # evaluation parameters
    parser.add_argument('--exp-times', default=10, type=int,
                        help='How many times of experiments')

    parser.add_argument('--classification', action='store_true',
                        help='Node classification task.')
    parser.add_argument('--clf-ratio', default="0.5",
                        help='The list for ratio of training data in the classification, separated by ,')

    parser.add_argument('--link-prediction', action='store_true',
                        help='Link prediction task.')
    parser.add_argument('--prop-pos', default=0.5, type=float,
                        help='proportion of positive edges for link prediction')
    parser.add_argument('--prop-neg', default=0.5, type=float,
                        help='proportion of negative edges for link prediction')
    parser.add_argument('--prop-neg-tot', default=1.0, type=float,
                        help='total proportion of negative edges for link prediction')
    parser.add_argument('--cached-fn', default='',
                        help='name of cached/to-be-cached graph file for link prediction task.')

    parser.add_argument('--reconstruction', action='store_true',
                        help='Network reconstruction task.')
    parser.add_argument('--k-nbrs', default=30, type=int,
                        help='K for knn in reconstruction')

    parser.add_argument('--clustering', action='store_true',
                        help='Vertex clustering task testing NMI.')
    parser.add_argument('--modularity', action='store_true',
                        help='Vertex clustering task testing modularity')
    parser.add_argument('--min-k', default=2, type=int,
                        help='minimum k for modularity')
    parser.add_argument('--max-k', default=30, type=int,
                        help='maximum k for modularity')
    args = parser.parse_args()

    return args

def load_embeddings(filename):
    fin = open(filename, 'r')
    node_num, size = [int(x) for x in fin.readline().strip().split()]
    vectors = {}
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split(' ')
        assert len(vec) == size+1
        vectors[vec[0]] = [float(x) for x in vec[1:]]
    fin.close()
    assert len(vectors) == node_num
    return vectors

def main(args):
    node_embeddings = load_embeddings(args.embedding_file)
    if args.label_file:
        labels = read_node_label(args.label_file)

    if args.modularity:
        print("Modularity")
        modularity(args, node_embeddings, args.min_k, args.max_k)

    if args.reconstruction:
        print("Graph reconstruction")
        reconstr(args, node_embeddings, args.k_nbrs)

    if args.clustering:
        print("Clustering")
        clustering(node_embeddings, labels, args.exp_times)

    if args.link_prediction:
        print("Link prediction")
        link_prediction(args.input, node_embeddings)

    if args.classification:
        X = list(labels.keys())
        Y = list(labels.values())
        print("Node classification")
        clf_ratio_list = args.clf_ratio.strip().split(',')
        result_list = {}
        train_ratio = np.asarray(range(1, 10)) * .1
        for clf_ratio in train_ratio:  # clf_ratio_list:
            result_per_test = []
            for ti in range(args.exp_times):
                clf = Classifier(vectors=node_embeddings, clf=LogisticRegression())
                myresult = clf.split_train_evaluate(X, Y, float(clf_ratio))
                result_per_test.append(myresult)
            result_list[clf_ratio] = result_per_test

        print('-------------------')
        for clf_ratio in train_ratio:
            print('Train percent:', clf_ratio)
            results = result_list[clf_ratio]
            for index, result in enumerate(results):
                print('Shuffle #%d:   ' % (index + 1), result)

            avg_score = defaultdict(float)
            for score_dict in results:
                for metric, score in score_dict.items():
                    avg_score[metric] += score
            for metric in avg_score:
                avg_score[metric] /= len(results)
            print('Average score:', dict(avg_score))
            print('-------------------')
        # print("\nmicro")
        # for i in range(exp_num):
        #     print("{}\t".format(result_list[i]["micro"]/args.exp_times), end='')
        # print("\nmacro")
        # for i in range(exp_num):
        #     print("{}\t".format(result_list[i]["macro"]/args.exp_times), end='')
        # print("\n")
        #     for nam in myresult.keys():
        #         if ti == 0:
        #             result[nam] = myresult[nam]
        #         else:
        #             result[nam] += myresult[nam]
        # for nam in result.keys():
        #     print("clf_ratio = {}, {}: {}".format(clf_ratio, nam, result[nam]/args.exp_times))
        # result_list += [result]

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    main(parse_args())
