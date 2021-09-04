import json
import os

from preprocessing.yeast.networks import get_undirected_net
from preprocessing.yeast.consts import RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH, GENERATED_BREITKREUTZ_ANNOTATIONS_PATH
from preprocessing.yeast.consts import RAW_MACISAAC_PDIS_DATA_PATH, GENERATED_MACISAAC_PDIS_PATH, PDI_MAPPER


def get_kpis(src_path=RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH,
             dst_path=GENERATED_BREITKREUTZ_ANNOTATIONS_PATH,
             net_type='biogrid', filter_by_net=False, force=False):
    dst_path = str(dst_path).format(net_type)
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            true_annotations_list = json.load(f)
            # Represent edges as tuples and not list
            return list(map(lambda e: tuple(e), true_annotations_list))
    else:
        net = get_undirected_net(net_type)
        genes = net.nodes

        true_annotations_list = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for entry in data[1:]:
            common_name1, sys_name1, common_name2, sys_name2, assay, method, action, modification, pub_source, ref, note = entry.replace('\n', '').split('\t')
            if net_type in ['biogrid', 'anat']:
                gene1 = common_name1
                gene2 = common_name2
            elif net_type == 'y2h':
                gene1 = sys_name1
                gene2 = sys_name2
            else:
                raise ValueError("Unsupported network type")

            if filter_by_net and (gene1 not in genes or gene2 not in genes):
                continue

            if method == 'high-throughput' and pub_source == 'BioGRID':
                if action == 'Bait-Hit':
                    edge = (gene1, gene2)
                elif action == 'Hit-Bait':
                    edge = (gene2, gene1)
                else:
                    raise Exception()

            # Do not include both direction edges
            if edge in true_annotations_list or (edge[1], edge[0]) in true_annotations_list:
                continue
            true_annotations_list.append(edge)

        with open(dst_path, 'w') as f:
            json.dump(true_annotations_list, f)

        return true_annotations_list


def get_pdis(src_path=RAW_MACISAAC_PDIS_DATA_PATH,
             dst_path=GENERATED_MACISAAC_PDIS_PATH,
             net_type='biogrid', filter_by_net=False, force=False):
    dst_path = str(dst_path).format(net_type)
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            true_annotations_list = json.load(f)
            # Represent edges as tuples and not list
            return list(map(lambda e: tuple(e), true_annotations_list))
    else:
        net = get_undirected_net(net_type)

        mapper = dict()
        mapper_path = str(PDI_MAPPER).format(net_type)
        with open(mapper_path, 'r') as f:
            data = f.readlines()
        for line in data:
            line_data = line.replace('\n', '').split('\t')
            if net_type == 'biogrid':
                mapper[line_data[0]] = line_data[1]
            elif net_type == 'y2h':
                mapper[line_data[1]] = line_data[0]
            else:
                raise ValueError("Unsupported network type")

        pdis = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for line in data:
            line_data = line.replace('\n', '').split('\t')
            source = line_data[0]
            targets = line_data[1:]
            if net_type == 'biogrid':
                for target in targets:
                    mapped_target = mapper.get(target)
                    real_target = mapped_target or target
                    if filter_by_net and source in net and real_target in net:
                        if (real_target, source) in pdis:
                            continue
                    if real_target and source != real_target:
                        pdis.append((source, real_target))
            elif net_type == 'y2h':
                mapped_source = mapper.get(source)
                for target in targets:
                    if filter_by_net and mapped_source in net and target in net:
                        if (target, mapped_source) in pdis:
                            continue
                    if mapped_source and mapped_source != target:
                        pdis.append((mapped_source, target))
            else:
                raise ValueError("Unsupported network type")

        pdis = list(filter(lambda e: (e[1], e[0]) not in pdis, pdis))

        with open(dst_path, 'w') as f:
            f.write(json.dumps(list(map(lambda x: (x[0], x[1]), pdis))))

    return pdis


if __name__ == '__main__':
    get_kpis(net_type='biogrid', force=True)
    get_pdis(net_type='biogrid', force=True)

    get_kpis(net_type='y2h', force=True)
    get_pdis(net_type='y2h', force=True)
