# Running the Project

Library versions used in this project:

```
Tensorflow Version: 1.14.0
Numpy Version: 1.20.3
Pandas Version 1.3.2
```

This project includes 5 Jupyter notebooks to show my thought process and the different iterations I’ve done for this project. 

Data were not included to keep the data confidential. Please add the raw CSV “Input.csv” and “output.csv” directly in this folder to run the code. 

# Methodology & Thoughts Process

## 1. Understanding the context of hearing loss and audiogram

My initial step consisted in learning a bit more about the data of the present challenge. After looking at the input/output CSV, it seems that we are working with audiograms. Notably, I found the *Healthy Hearing* website and it vulgarized really well what is hearing loss, how an audiogram is conducted and assesses its results (input of this challenge), and how hearing aids settings are defined (output of this challenge). 

![Untitled](Zepp%20-%20Rea%209d81f/Untitled.png)

Specific links include:

1. Hearing loss: [https://www.healthyhearing.com/report/41775-Degrees-of-hearing-loss](https://www.healthyhearing.com/report/41775-Degrees-of-hearing-loss)
2. Audiogram: [https://www.healthyhearing.com/report/52516-The-abc-s-of-audiograms](https://www.healthyhearing.com/report/52516-The-abc-s-of-audiograms)
3. Hearing aid settings: [https://www.healthyhearing.com/report/53222-Hearing-aid-settings-customize](https://www.healthyhearing.com/report/53222-Hearing-aid-settings-customize)

Notably, this clarified the intention of the model developed: the theoretical fitting formula aims at maximizing speech intelligibility while ensuring the overall loudness does not exceed the loudness perceived by a normal-hearing person. 

## 2. Exploring the dataset and quality checks

In a second step, I explored the datasets and checked the data quality to ensure we had no missing data, abnormal values or extreme data. 

**Quality checks and explorations:**

- Checked the minimum and maximum values of the columns
    - Audiogram tests vary between -10 and [95-120] decibels which concords with information gathered around audiograms
    - Gender is either 1 or 2
    - Age is between 7yo and 85yo
    - Experience with earing aids and compressor speed are binary 0 or 1
    - Hearing aids settings varies between 2 and 57. The value “2” seems to be the default value when no earing aids are needed for this specific frequency.
- Checked if there were missing values
- There are 330 participants (based on their ID) that are multiple times in the input dataset.

**** Inputs and outputs size does not match. ****

I realized that the size of the input (n=2999) and output (n=29 999) files does not match. After looking at the data, it seems like the rows of the input file are directly linked to the firsts rows of the outputs — i.e. the output is not cropped to the same length but still matches the input. 

Indeed, when no earing aids are needed and the audiogram results show that the patient has a normal hearing ability for a frequency, the corresponding hearing aids settings of the row are set to the value 2. 

Thus, I cropped the output dataset to get the same length as the input dataset.

## 3. The problem

Next, I started working on the actual problem at hand by defining my input and output formats, followed by the pro 

- Inputs: array of value (tabular data) with features of different amplitude [continuous values] , some like gender are categorical [men/women] and others are binary [true/false]
- Outputs: array of continuous values representing the settings of the hearing aids device

In this problem, we are trying to predict multiple continuous outputs (i.e. predicting more than one numeric value). This sort of task is referred to as **multiple-output regression**, i.e. two or more outputs are required for each input sample, and the outputs are required simultaneously.

## 4. Looking at the literature

Now that I have a good grasp of the problem, I looked more at the scientific literature around this topic. Notably, I found the following article that tackles a very similar challenge. In this article, the authors are using a Multi-Layer perception to predict hearing aids settings based on audiogram tests (without inputs such as gender, age, etc.)

- Mondol, S. I. M. M., & Lee, S. (2019). A machine learning approach to fitting prescriptions for hearing aids. *Electronics*, *8* (7), 736.

This article led me to look at the NAL-NL1 and NAL-NL2 prescription procedures for tunning the hearing aids. Notably, NAL-NL2 takes the profile of the hearing-aid user’s age, gender, experience, language, and compressor speed into consideration — which is the same inputs as this challenge. For example: *“The analysis revealed that female hearing aid users, irrespective of degree of hearing loss and experience with amplification preferred less gain (2 dB, on average) than male hearing aid users”* (Keidser & Dillon, 2006)

- Keidser, G., Dillon, H., Flax, M., Ching, T., & Brewer, S. (2011). The NAL-NL2 prescription procedure. *Audiology research,* *1* (1), 88-90.

## 5. Choosing a neural network & Metrics

I chose to use a Multi-Layer Perceptron (against CNN or RNN) since MLP [[source](https://machinelearningmastery.com/when-to-use-mlp-cnn-and-rnn-neural-networks/)]:

1. Works best with tabular data (like the input given)
2. Works well for regression prediction problems and multiple outputs (like the output)
3. Is good to learn how to map inputs to outputs (like the stated problem)

**Applying neural networks to multi-output regression problems**

Multi-output regression can be supported by neural networks by specifying the number of target variables (number of settings of the hearing aids) as the number of nodes in the output layer. 

**Activation functions & weight initialization**

I used the *Rectified Linear Unit* (ReLu) as the activation function for the input layer and the hidden layers. ReLu is commonly used for hidden layers, especially since it’s simple to implement. With ReLu it is a good practice to use a “*He Normal”* weight initialization. [[source](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)]

For the output layer, since we are working with a regression outcome, I’ve used the linear activation function

**Metrics: Mean Absolute Error**

Since we are looking at settings of the hearing aids, I like to think that this model would be used directly by audiologists or patients to tune the hearing aids. In this case, it would be easier for them to understand the model if the units of the error score match the units of the hearing aids, therefore more intuitive for clinicians and patients. 

**Training and Testing set**

I used a repeated K-Fold (10 splits) to train and test the model. The average result of the K-fold represents the overall performance of the model. 

## 6. Baseline, Models and tunning

**Baseline (see Jupyer file “2. Baseline Model”)**

As seen in Keidser & Al. (2011) article, the initial NAL-NL1 is based only on the audiology test, which was my starting point to define a baseline. 

In this baseline, I used 7 inputs nodes (7 features), 1 hidden layer of 10 nodes and 7 outputs notes (7 parameters of the hearing aids). 

**Model 1 (see Jupyter file “3. With demographics V1”)**

In this model, I added the demographics without any transformation. 

**Model 2 (see Jupiter file “4. With demographics V2”)**

In this model, I used the One-hot-encoder to change the gender into a separate variable [gender_1, gender_2]. 

**Hyperparameter tuning (see Jupyter file “5. Tunning”)**

In this last step, I try different 1) number of hidden layers, 2) different nodes per hidden layers and 3) two different optimizers. Having a hard time installing a library to add a random search and wanting the work more on the model itself, I ended up adapting my previous function into a flexible model to try a few combinations.

### Model comparison - **Results**

|  | Features | MAE - Mean (STD) |  |
| --- | --- | --- | --- |
| Baseline | Results from the audiology test only (7) | MAE 0.982 (0.057) | Input layer: 7 nodes
Hidden layer: 1 (10 nodes)
Output layer: 7 nodes |
| With demographics (V1) | All inputs (11), no transformation | MAE: 0.947 (0.064) | Input layer: 11 nodes
Hidden layer: 1 (10 nodes)
Output layer: 7 nodes |
| With demographics (V2) | All inputs, One hot encoding of gender | MAE: 0.930 (0.080) | Input layer: 12 nodes
Hidden layer: 1 (10 nodes)
Output layer: 7 nodes |
| Random Search (10x) | All inputs, One hot encoding of gender | MAE: 0.736 (0.050) | Best Model:
Input layer: 12 nodes
Hidden layers: 3 [48, 48, 12 nodes]
Output layer: 7 nodes |

Each interaction did improve slightly the MAE of the model with our best model reaching an MAE of `0.736` showing that the tunning of each hearing aid parameter is only off by +/- 0.736. 

## 7. Next & Improvement

1. Using Jupyter Notebook is quick and easy to set up for data exploration, trying out models, keeping track our my different iterations, etc. Which was perfect for a quick weekend challenge. Having more time, I would have created more transferable .py files. 
2. I am currently using the same function in multiple notebooks - since I wanted to keep track of my iterations. In a production or research context, I would have created those functions in a separate .py file and imported the function in the notebook.
