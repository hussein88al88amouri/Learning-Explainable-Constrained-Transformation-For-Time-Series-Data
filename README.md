# Learning Explainable Constrained Transformation For Time Series Data

**Author:** Hussein El Amouri  
**Affiliation:** ICube, University of Strasbourg  
**Email:** alamourihusein@gmail.com

This repository contains the implementation of work prposed during the three years of PhD at Icube Laboratiry.
The work encamposes two different contributions:
- CDPS: Constrained DTW Preserving shapelets [1]. This approach aims at learning a transformation by guiding the model using expert knowledge, while approximating DTW similarity between time series in transformed time series space to account for the distortions. The work also includes a poster and the paper with supplementary materials.
- SCE: Shapelet Cluster Explanation [2]. This approach offeres a way to explain the output of clustering results based on the learned shapelets using Inforamtion Gain, different expalanation strategies have been proposed/

## Reference
[1] Hussein El Amouri, Thomas Lampert, Pierre Gançarski, Clément Mallet, CDPS: Constrained DTW-Preserving Shapelets. European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases 2022. [DOI](https://doi.org/10.1007/978-3-031-26387-3_2) [Link to the paper](https://hal.archives-ouvertes.fr/hal-03736948).
[2] Hussein El Amouri, Thomas Lampert, Pierre Gançarski, Clément Mallet, Constrained DTW preserving shapelets for explainable time-series clustering. Pattern Recognition, 2023. [DOI](https://doi.org/10.1016/j.patcog.2023.109804),  [Link to the paper](https://hal.science/hal-04171112v1/document).
 
## Instructions
To use the toolbox, download UCR datasets [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), containing both Multivariate and Univariate datasets. The scripts expect datasets in the tsfile format provided by the UCR archive. In the `scripts` subdirectory, update the `dataPath` variable to the absolute path of the datasets directory.

The `scripts` subdirectory includes scripts to:
- Generate constraints: `GenerateConstraints.sh`
- Initialize the model: `InitializeModelParam.sh`
- Transform the dataset using the shapelet transform: `TransformDataSet.sh`
- Train the model using various conditions: `Train_CLDPS_pytorch_<conditions>.sh`
  - Conditions include: `Noconstraints`, `diffAlphaGamma`, `diffConstraints`, `diffshapelets`.

This work is related to TSCC: Time-Series Constrained Clustering. The TSCC repository can be found [here](https://icube-forge.unistra.fr/lampert/TSCC/-/tree/master/). In the `bin` subdirectory, there is a script to prepare the transformed data for the correct format used by TSCC. Update the `datasets` variable to point to the dataset under study, and `embeddingpathdirs` to point to the absolute path of the transformed data.

Note: The `bin` subdirectory includes an implementation of CopKmeans using DBA averaging for multivariate datasets. It is an extension of the implementation provided by [Behrouz-Babaki/COP-Kmeans](https://github.com/Behrouz-Babaki/COP-Kmeans).

## Algorithms
The following implementations are included:
- **Subdirectory:** FeatTS
  - **Name:** Feature-driven Time Series Clustering
  - **Language:** Python
  - **URL:** [FeatTS Repository](https://github.com/protti/FeatTS)
  - **Publication:** B. Tiano, D., Bonifati, A., & Ng, R. (2021, May). Feature-driven Time Series Clustering. In 24th International Conference on Extending Database Technology, EDBT 2021.

## Funding
This work was supported by the HIATUS (ANR-18-CE23-0025) and HERELLES (ANR-20-CE23-0022) ANR projects. We thank Nvidia Corporation for donating GPUs, and the Centre de Calcul de l’Université de Strasbourg for access to the GPUs used for this research.


