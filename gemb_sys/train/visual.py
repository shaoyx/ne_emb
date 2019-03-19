import matplotlib.pyplot as plt
import sys
from sklearn.metrics.pairwise import cosine_similarity
import pickle

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

if __name__ == "__main__":
    input = sys.argv[1]
    output = sys.argv[2]

    vectors = load_embeddings(input)
    x = []
    keys = list(vectors.keys())
    size =len(keys)
    for i in range(size):
        for j in range(i+1, size):
            v1 = vectors[keys[i]]
            v2 = vectors[keys[j]]
            sim = cosine_similarity([v1],[v2])
            sim_val = sim[0][0]
            x.append(sim_val)

    # with open("sim.cora", "wb") as f:
    #     pickle.dump(x, f)

    # with open("sim.cora", "rb") as f:
    #    x = pickle.load(f)
    # print(len(x))
    # print(x[0])
    # the histogram of the data
    plt.hist(x, 10000) #, 10000, density=True, facecolor='g', alpha=0.75)

    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, -1, 1])
    # plt.grid(True)
    plt.savefig(output+".pdf")