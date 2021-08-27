from preprocessing.consts import SOURCES_DIR

YEAST_SOURCES_DIR = SOURCES_DIR / 'yeast'
YEAST_GENERATED_SOURCES_DIR = YEAST_SOURCES_DIR / 'generated'

RAW_YEAST_BIOGRID_PATH = YEAST_SOURCES_DIR / 'BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-4.4.199.tab3.txt'
GENERATED_YEAST_BIOGRID_NET_PATH = YEAST_GENERATED_SOURCES_DIR / 'biogrid_net.gpickle'

RAW_YEAST_Y2H_PATH = YEAST_SOURCES_DIR / 'Y2H_union.txt'
GENERATED_YEAST_Y2H_NET_PATH = YEAST_GENERATED_SOURCES_DIR / r'y2h_union_net.gpickle'

RAW_BREITKREUTZ_ANNOTATIONS_DATA_PATH = YEAST_SOURCES_DIR / 'Breitkreutz_kpis_annotations.txt'
GENERATED_BREITKREUTZ_ANNOTATIONS_PATH = YEAST_GENERATED_SOURCES_DIR / 'breitkeurtz_annotations_{}.json'

RAW_HOLSTEGE_EXPRESSIONS_DATA_PATH = YEAST_SOURCES_DIR / 'Holstege_experiments_expressions.cdt'
GENERATED_HOLSTEGE_EXPRESSIONS_PATH = YEAST_GENERATED_SOURCES_DIR / 'holstege_expressions_{}.json'

RAW_MACISAAC_PDIS_DATA_PATH = YEAST_SOURCES_DIR / 'macisacc_kpis_orfs_by_factor_p0.001_cons2.txt'
GENERATED_MACISAAC_PDIS_PATH = YEAST_GENERATED_SOURCES_DIR / 'macisaac_pdis_{}.json'

GENERATED_EDGES_TO_DIRECT_PATH = YEAST_GENERATED_SOURCES_DIR / r'edges_to_direct_in_{}.json'

GENERATED_YEAST_DIRECTED_NET_PATH = YEAST_GENERATED_SOURCES_DIR / r'directed_{}_net.gpickle'

PDI_MAPPER = YEAST_SOURCES_DIR / r'{}_pdis_mapper.tsv'
