import json
import os

from preprocessing.yeast.biogrid.read_biogrid import get_biogrid_network
from preprocessing.yeast.y2h.read_y2h_union import get_y2h_union_network
from preprocessing.yeast.consts import RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH, GENERATED_BREITKREUTZ_ANNOTATIONS_PATH
from preprocessing.yeast.consts import RAW_MACISAAC_PDIS_DATA_PATH, GENERATED_MACISAAC_PDIS_PATH, PDI_MAPPER


def get_kpis(src_path=RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH,
             dst_path=GENERATED_BREITKREUTZ_ANNOTATIONS_PATH,
             net_type='biogrid', filter_by_net=True, force=False):
    dst_path = str(dst_path).format(net_type)
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            true_annotations_list = json.load(f)
            # Represent edges as tuples and not list
            return list(map(lambda e: tuple(e), true_annotations_list))
    else:
        if net_type == 'biogrid':
            net = get_biogrid_network()
        elif net_type == 'y2h':
            net = get_y2h_union_network()
        else:
            raise ValueError("Unsupported network type")
        genes = net.nodes

        true_annotations_list = []
        with open(src_path, 'r') as f:
            data = f.readlines()
        for entry in data[1:]:
            common_name1, sys_name1, common_name2, sys_name2, assay, method, action, modification, pub_source, ref, note = entry.replace('\n', '').split('\t')
            if net_type == 'biogrid':
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
            if edge in true_annotations_list:
                continue
            true_annotations_list.append(edge)

        with open(dst_path, 'w') as f:
            json.dump(true_annotations_list, f)

        return true_annotations_list


def get_pdis(src_path=RAW_MACISAAC_PDIS_DATA_PATH,
             dst_path=GENERATED_MACISAAC_PDIS_PATH,
             net_type='biogrid', filter_by_net=True, force=False):
    if os.path.isfile(dst_path) and not force:
        with open(dst_path, 'r') as f:
            true_annotations_list = json.load(f)
            # Represent edges as tuples and not list
            return list(map(lambda e: tuple(e), true_annotations_list))
    else:
        if net_type == 'biogrid':
            net = get_biogrid_network(force=False)
        elif net_type == 'y2h':
            net = get_y2h_union_network(force=False)
        else:
            raise ValueError("Unsupported network type")

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
                    if target in net:
                        real_target = target
                    elif mapped_target in net:
                        real_target = mapped_target
                    else:
                        continue
                    if filter_by_net and source in net and real_target in net:
                        if (real_target, source) in pdis:
                            continue
                    if source != real_target:
                        pdis.append((source, real_target))
            elif net_type == 'y2h':
                mapped_source = mapper.get(source)
                for target in targets:
                    if mapped_source in net and target in net and mapped_source != target:
                        if (target, mapped_source) in pdis:
                            continue
                        else:
                            pdis.append((mapped_source, target, 1))
            else:
                raise ValueError("Unsupported network type")

        with open(dst_path, 'w') as f:
            f.write(json.dumps(list(map(lambda x: (x[0], x[1]), pdis))))


if __name__ == '__main__':
    get_kpis(net_type='biogrid', force=True)
    get_pdis(net_type='biogrid', force=True)

    get_kpis(net_type='y2h', force=True)
    get_pdis(net_type='y2h', force=True)
