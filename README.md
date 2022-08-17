# About the Project  
A generative adversarial network (GAN) is an implicit generative model made up of two neural networks. GANs have a distinctive training architecture that is designed to create examples that reproduce a target distribution. These models have been applied successfully in high-dimensional domains such as natural image generation and processing. Much less research has been reported for applications with low dimensional distributions, where properties of GANs may be better identified and understood. One such area in finance, is the use of GANs for estimating value-at-risk (VaR). Through this financial application, this dissertation introduces readers with the concepts and practical implementations of GAN variants to generate synthetic portfolio returns. Large portions of the discussions should be accessible to anyone who has an entry level statistics course. It is aimed at data science or finance students looking to better their understanding of GANs and the potential of these models for other financial applications. Five GAN loss variants are introduced and three of these models are practically implemented to estimate VaR. The GAN estimates are compared to more traditional VaR estimation techniques and all models are backtested.

# Objective's of the Repo
This repo contains the code used to complete my minor dissertation. The different GAN models used as well as all the helper functions are provided so user's can experiment with different hyper-parameters and get a better understanding of how to train GANs in practice. 

# How to use this repo 
There are various sections to the project. All the final GAN model checkpoints are included under the 'checkpoints' file. These can be loaded and used to generate data using the various testing files located in the 'testing' directory. Historical datasets used for training the GANs are located in 'data quantiles', this is included for accessing historical VaR in the testing files. 

## Step 1: Updating Directory Information

## Step 2: Training a GAN Model (Optional)

## Step 3: Loading a GAN model checkpoint 
