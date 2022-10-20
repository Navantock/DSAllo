# DSAllo
Deep Surface Allosteric Sites Predictor


Author：Hongfei Wu

E-mail: 1900011764@pku.edu.cn

Institution: College of Chemistry and Molecular Engineering of Peking University

This project is a allosteric sites predictor on the base of DS-GCN.

The full article is available at "Study on the Prediction of Protein Allosteric Sites Using Graph Neural Networks.pdf" in this directory.

## Requirement

#### 1. Biopython

> [Biopython · Biopython](https://biopython.org/)

#### 2. torch ; torch_geometric

#### 3. matplotlib

#### 4. msms

> [mgltools (scripps.edu)](https://ccsb.scripps.edu/mgltools/#msms)

## Instruction

If you are going to use our model to predict the allosteric sites of  a protein structure, you can simply run the commad in your shell at current directory:

```shell
python main.py --pdb_id xxxx
```

The prameters are shown as below:

**--f**: the pdb file used for prediction

**--pdb_id**: the protein pdb_id used for prediction. Note that if you have used pdb_id, the program will download the corresponding pdb file from RCSB and param --f will be aborted.

**--p**: the model for prediction. We suppose you use the default "DS_CycleGCN" as predictor.

**--pf**: the model parameters file. If you use the default model, there is no need to change this param.

**--v**: visualization tag. The predictor will generate a picture to visualize the results if use 1 as --v, else not.

**--vis_mode**: visualization mode. Use "show" to immediately show the results after calculated, else 

the program will save the picture in the ./results
