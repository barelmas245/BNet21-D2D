from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.annotations import get_kpis

if __name__ == '__main__':
    net = get_biogrid_network()
    true_annotations = get_kpis()

    lines = []
    for e in net.edges:
        id1 = e[0]
        id2 = e[1]
        conf = net[id1][id2]["weight"]
        direction = 1

        e_type = 'UNDIRECTED_KEGG'
        if (id1, id2) in true_annotations:
            e_type = 'TRUE_KEGG'
        elif (id2, id1) in true_annotations:
            e_type = 'FALSE_KEGG'
        print(len(lines))
        lines.append('\t'.join([id1, id2, str(conf), str(direction), e_type]) + '\n')

    with open(r'C:\git\BNet21-D2D\results\single_unified_experiment\biogrid_fast.net', 'w') as f:
        f.writelines(lines)
