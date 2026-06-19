"""Music Genre Classification — source package.

Subpackages:
    data        - GTZAN validation, stratified splitting, dataset loaders
    features    - librosa feature extraction (classic) + mel-spectrograms (CNN)
    models       - model definitions (classic.py, cnn.py)
    training    - training entry points (train_classic.py, train_cnn.py)
    evaluation  - unified metrics + classic-vs-CNN comparison
    api         - FastAPI serving app
"""
