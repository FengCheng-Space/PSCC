# PSCC
-----
Positive-Sample contrastive cluster （PSCC）: a deep learning model for Single Cell RNA-seq data cluster. The PSCC algorithm adopted a saimese contrastive learning network that learns and reinforces features using positive-sample data. We employ a zero-inflated negative binomial distribution model for denoising and dimensionality reduction of scRNA-seq data during feature learning. Moreover, we estimate and optimize the loss of Branch network contrastive learning and Spatial mapping learning to improve the PSCC method’s feature learning and reinforcement capabilities on scRNA-seq data.



# Data availability
-----
The example dataset saved in the location  "/Example_data/Quake_10x_Bladder.h5" and you need to unzip the folder "Quake_10x_Bladder.zip" to get .h5 data. Other datasets used in the experiments are available at https://github.com/xuebaliang/scziDesk and https://github.com/WHY-17/scDSSC.

# Architecture
-----

![model](https://github.com/FengCheng-Space/PSCC/blob/main/Architecture/Figure1.jpg)


# Quick start
-----
We provided an example dataset:"Quake_10x_Bladder" in the <a herf="https://github.com/FengCheng-Space/PSCC/tree/main/Example_data">folder as a defult datasets. You just run the following code in your command lines:
"python main.py"

# Requirement
-----

python >= 3.8

pytorch  >=2.0.0

h5py >=3.7.0

scanpy >= 1.8.1

sklearn >=1.0.2


