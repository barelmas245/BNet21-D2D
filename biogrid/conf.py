# Confidence scores for different types of experiment types from BIOGRID
# The scores are based on Gitter et al. 2010
BIOGRID_EXPERIMENT_TYPES_CONFIDENCE_SCORES = {
    'Affinity Capture-Luminescence': 0.5,
    'Affinity Capture-MS': 0.5,
    'Affinity Capture-RNA': 0.7,
    'Affinity Capture-Western': 0.5,
    'Biochemical Activity': 0.5,
    'Co-crystal Structure': 0.99,
    'Co-fractionation': 0.7,
    'Co-localization': 0,
    'Co-purification': 0.7,
    'Dosage Growth Defect': 0,
    'Dosage Lethality': 0,
    'Dosage Rescue': 0,
    'Far Western': 0.5,
    'FRET': 0.7,
    'PCA': 0.3,
    'Phenotypic Enhancement': 0,
    'Phenotypic Suppression': 0,
    'Protein-peptide': 0.7,
    'Protein-RNA': 0.3,
    'Reconstituted Complex': 0.3,
    'Synthetic Growth Defect': 0,
    'Synthetic Haploinsufficiency': 0,
    'Synthetic Lethality': 0,
    'Synthetic Rescue': 0,
    'Two-hybrid': 0.3,
    'Proximity Label-MS': 0  # Additional experiment type not mentioned in Gitter et al. 2010
}

# Use only physical interactions (and not genetic ones)
ONLY_PHYSICAL = True
