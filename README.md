# astro_FDIDWT

This repo lists the scripts for the paper "Identification of 4FGL uncertain sources at Higher Resolutions with Inverse Discrete Wavelet Transform". Please run the codes step by step. More details can be found in our paper.

## File Tree

`|-- FD_ASE.py`
`|-- README.md`
`|-- attribute_params.py`
`|-- datasets`
`|   |-- 4FGL_DR3_Data_A.xlsx`
`|   |-- 4FGL_DR3_Data_B.xlsx`
`|-- models.py`
`|-- step0_preprocessing_4FGL_data.ipynb`
`|-- step1_split_data.ipynb`
`|-- step2_analysis_FDIDWT_iwt.m`
`|-- step2_attributes_analysis.ipynb`
`|-- step3_training.ipynb`
`|-- utils.py`



- `FD_ASE.py`: the functions of FDASE algorithms
- `README.md`: this file
- `attribute_params.py`: the settings of attributes after attribute analysis
- `datasets`: the directory for data, containing the original 4FGL_DR3 data (13 attributes), and the generated data after running the codes
- `models.py`: the definitions of the neural networks, i.e., Multi-layer Perceptron (MLP) and the proposed MatchboxConv1D model
- `step0_preprocessing_4FGL_data.ipynb`: the step to preprocess the original 4FGL_DR3 data, e.g., process the NaN values
- `step1_split_data.ipynb`: the first step to split the data into training/validation/test sets
- `step2_analysis_FDIDWT_iwt.m`: the MATLAB script to perform Inverse Discrete Wavelet Transform (IDWT)
- `step2_attributes_analysis.ipynb`: the methods to analyze the attributes, including Attribute Importance (RF), PCA, and FDASE
- `step3_training.ipynb`: the Machine Learning (ML) methods and Neural Network (NN) models to accomplish the two missions. The results will be written into a new directory named `RESULTS`.
- `utils.py`: some tools

## Experimental Steps

1. Run `step0_preprocessing_4FGL_data.ipynb` to prepare the data
2. Run `step1_split_data.ipynb` to obtain the training/validation/test sets. In the paper we obtained 10 splits for the uncertainties of the results.
3. Run `step2_attributes_analysis.ipynb` to dig the information of attributes. The results were recorded to `attribute_params.py`. 
4. Run `step2_analysis_FDIDWT_iwt.m` to obtain the IDWT data for the next step. (We are planning to implement IDWT in Python)
5. Run `step3_training.ipynb` to perform classification. The results can be found in a new directory named `RESULTS`.

## Cite

If this repo helps you, please cite us.