## About Model

**Purpose**

* This model was developed to practice development of ncRNA sequence generation
* It is highly experimental, and should not be used in any real-world environments where accuracy is paramount

**Methodology**

* We utilize GNN architecture to predict RNA sequences which would fold towards a given structure
* The idea is that a researcher might want to have a ncRNA with a specific structure that will fold into a desired shape
* They might know what structure would make this shape, but they do not know the sequence that makes that structure

**Training Data**

* Trained on RFAM seed alignments RF00001–RF00500 (~500 RNA families, Stockholm format)
* Data sourced from the EBI RFAM REST API; see `COLAB_SETUP.md` for download instructions

**Training Details**

* Teacher forcing ratio is linearly annealed from 1.0 (epoch 1) to 0.1 (final epoch) to reduce exposure bias
* Early stopping with patience of 15 epochs prevents overfitting
* Train/validation split is a seeded random 80/20 split across all families

**Usage**

* See `COLAB_SETUP.md` for step-by-step instructions to train and run inference on Google Colab
