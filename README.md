# RMTPP-pytorch
## Material in this file:
      Introduction
      Preparation and How to Run Code
      Model Description
      Result
      Acknowledgement and Reference
### Introduction
This repository is the implementation of model described in paper Du, Nan, et al. [“Recurrent Marked Temporal Point Processes.”](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process) by pytorch. This repository also use the same data as the data used in the paper. The rest of this section is a recap of the paper.

Discrete events with different types, which is defined as Marked Temporal Point Process, is usually generated in real life setting. Some past events may increase or decrease the occurrence probability of other kinds of event in the future. For example, in health care system having some type of medicines may cause a lower probability of some specific disease, and in social media like twitter some tweets may cause an increasing number of tweets of the same topic in a short time. The Recurrent Marked Temporal Point Process (RMTPP) model is robust to predict the next event time and next event type. That is, given a stream of past event times t and past event types k of form <pre>{(k<sub>1</sub>, t<sub>1</sub>), (k<sub>2</sub>, t<sub>2</sub>), … (k<sub>n</sub>, t<sub>n</sub>)}, where 1,2,n represents the 1st, 2nd, nth event</pre>, it can predict the next type and type <pre>(k<sub>n+1</sub>, t<sub>n+1</sub>)</pre>

### Preparation and How to Run the Code
You can run the code on your own computer or on python notebooks like Google Colab, Anaconda Jupyter Notebook. It is recommended to run the script on google colab (https://colab.research.google.com/) or similar python notebooks if you do not have a dedicated GPU on your computer. 

1. To run the program on your computer, please make sure that you have the following files and packages being downloaded.<br />
- Python3: you can download through the link here: https://www.python.org/ </pre>
- Numpy: you can dowload it through command <pre>pip install numpy</pre>
- Scikit-Learn: you can download it through command <pre>pip install sklearn</pre>
- Scipy: you can download it through command <pre>pip install scipy</pre>
- Pytorch: pytorch installation is more complicated than the package described above. You can go to https://pytorch.org/get-started/locally/ for more information. If you still cannot install it on windows computer through pip, you can download Anaconda first and then download the pytorch through method described here: https://dziganto.github.io/data%20science/python/anaconda/Creating-Conda-Environments/ <br /><br />


2. You can also run the code on Google Colab. You first need to connect this repository to Colab by the following code:<br /><pre>!git clone https://github.com/Hongrui24/RMTPP-pytorch</pre> Then cd to this directory by <pre>!cd RMTPP-pytorch</pre>

3. To train the model, you can type the following for guide: <pre>!python train.py --help</pre> Sample command line can be <pre>!python train.py --lr 0.03 --epochs 500 --data hawkes</pre><br />
To test the trained model, you can type the following for guide: <pre>!python test.py --help</pre> Sample command line can be <pre>!python test.py --data hawkes</pre>

After done the test, you can find the result plots in the same file named by "result.png". 

### Model Description
This is a recap of the model described in the paper:

This model is trained by Backpropogation Through Time (BPTT) method. That is, for each event stream {(k<sub>j</sub>, t<sub>j</sub>)} we take b consecutive events as input into the model, and unroll the model for b steps through time. However, we will use the whole history of event stream to make prediction for hawkes and self-correcting.  
We input the jth event (k<sub>j</sub>, t<sub>j</sub>) into the jth layer of the model. The event type, which is in one-hot representation, is embeded into a single value. <pre> y<sub>j</sub> = W<sub>e</sub>·k<sub>j</sub>+b </pre> We extract the inter-event duration from the time t<sub>j</sub> by making <pre> d<sub>j</sub> = t<sub>j</sub> - t<sub>j-1</sub>, and d<sub>0</sub> = 0 </pre> Then, we feed the type value and inter-event duration into the jth layer of the RNN cell. In order to prevent the dying relu problem, all weights are initialized with positive values. We update the hidden layer of the RNN by <pre> h<sub>j</sub> = relu(W<sub>y</sub>·y<sub>j</sub> + W<sub>t</sub>·d<sub>j</sub>+W<sub>h</sub>·h<sub>j-1</sub> + b)</pre> Then, we get the predict event type probability through softmax method. We can also get the time intensity and conditional density of time by <pre>λ*(t) = exp(V·h<sub>t</sub> + w*d<sub>t</sub> + bias)</pre> <pre>f(t) = exp(V·h<sub>t</sub> + w*d<sub>t</sub> + bias + (exp(V·h<sub>t</sub>+ bias)-exp(V·h<sub>t</sub> + w*d<sub>t</sub> + bias)/w)</pre>For training process, we minimize the negative log likelihood of the time, and we use the expectation to predict the next time. 



### Model Test Results

We test our model with the data used in the paper's generative experiments of predicting Hawkes and Self-correcting. The results show that the model implemented by pytorch is able to get the similar result as the result described in the paper after the model is trained with 500 epochs with 0.03 learning rate. In each of the picture below the graphs are prediction on inter-event duration, intensity, and Root-Mean-Squared-Error accordingly. <br />
Test on Hawkes:<br />
![Hawkes](https://user-images.githubusercontent.com/54515153/84570792-a630c800-ad5d-11ea-972e-a809f0865add.png)<br />
Comparing to the results on the paper:
![Hawkes](https://user-images.githubusercontent.com/54515153/85622693-3c55cf80-b635-11ea-96c9-3d43b0526556.JPG)<br />
![self-correcting](https://user-images.githubusercontent.com/54515153/84570795-a9c44f00-ad5d-11ea-9b71-30632793f9b4.png)<br />
Comparing to the results on the paper:
![Self-correcting](https://user-images.githubusercontent.com/54515153/85622697-3d86fc80-b635-11ea-99f1-d6d5835e642c.JPG)
Test on Self-correcting:<br />

### Acknowledgement
The model is built by Hongrui Lyu, supervised by of Hyunouk Ko and Xiaoming Huo. This repository is built upon the model described in paper Du, Nan, et al. [“Recurrent Marked Temporal Point Processes.”](https://www.kdd.org/kdd2016/subtopic/view/recurrent-temporal-point-process). We also use a similar pytorch implementation by [Yunxuan Xiao](https://github.com/woshiyyya/ERPP-RMTPP) to debug the model. 
