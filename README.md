# RMTPP-pytorch
## Material in this file:
      Introduction
      Preparation and How to Run Code
      Model Description
      Result
      Acknowledgement and Reference
### Introduction
This repository is the implementation of model described in paper Du, Nan, et al. “Recurrent Marked Temporal Point Processes.” by pytorch. This repository also use the same data as the data used in the paper. 

### Preparation and How to Run the Code
You can run the code on your own computer or on python notebooks like Google Colab, Anaconda Jupyter Notebook. It is recommended to run the script on google colab (https://colab.research.google.com/) or similar python notebooks if you do not have a dedicated GPU on your computer. 

1. To run the program on your computer, please make sure that you have the following files and packages being downloaded.<br />
- Python3: you can download through the link here: (https://www.python.org/) </pre>
- Numpy: you can dowload it through command <pre>pip install numpy</pre>
- Scikit-Learn: you can download it through command <pre>pip install sklearn</pre>
- Scipy: you can download it through command <pre>pip install scipy</pre>
- Pytorch: pytorch installation is more complicated than the package described above. You can go to (https://pytorch.org/get-started/locally/) for more information. If you still cannot install it on windows computer through pip, you can download Anaconda first and then download the pytorch through method described here: (https://dziganto.github.io/data%20science/python/anaconda/Creating-Conda-Environments/) <br /><br />


2. You can also run the code on Google Colab. You first need to connect this repository to Colab by the following code:<br /><pre>!git clone https://github.com/Hongrui24/RMTPP-pytorch</pre> Then cd to this directory by <pre>!cd RMTPP-pytorch</pre>

3. To train the model, you can type the following for guide: <pre>!python train.py --help</pre> Sample command line can be <pre>!python train.py --lr 0.03 --epochs 500 --data hawkes</pre><br />
To test the trained model, you can type the following for guide: <pre>!python test.py --help</pre> Sample command line can be <pre>!python test.py --data hawkes</pre>

After done the test, you can find the result plots in the same file named by "result.png". 

