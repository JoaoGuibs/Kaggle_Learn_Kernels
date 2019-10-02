# Kaggle_Learn_Kernels
Set of notebooks prepared for google colab that trains on different sets of data-sets that were taken from the Kaggle Website.

## Datasets used:
  * Ships in Satellite Images (https://www.kaggle.com/rhammell/ships-in-satellite-imagery)
  * Simulated and real lunar soil images (https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset)

#### Installation instructions:
First we need to install Miniconda by:

* Download the Miniconda installer file [from here](https://docs.conda.io/en/latest/miniconda.html)
* Install it (e.g., on Linux 64 bit use the command: bash Miniconda3-latest-Linux-x86_64.sh
* Change directory to the main folder and then create the conda environment by:
  * ``` conda env create -f environment.yml ```
  * ``` conda activate Kaggle_Kernels ```

If everything proceeded without error, we can open the jupyter notebook by:

* ```jupyter notebook ```

The .ipynb file can then be opened and edited by using the Jupyter's GUI, for simplicity.

#### Addittional Information:
If the models need to be trained, then a GPU might be necessary, therefore additional packages and the correct drivers might be needed to be installed:
* tensorflow-gpu (install through conda)
* CUDA [from here](https://developer.nvidia.com/cuda-downloads)
* CuDNN [from here](https://developer.nvidia.com/cudnn) (requires account)