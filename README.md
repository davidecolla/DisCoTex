## Overview
This repository contains the source code for the baseline evaluation for the [DisCoTEX (Assessing DIScourse COherence in Italian TEXts)](https://sites.google.com/view/discotex/home) shared task within the [EVALITA 2023](https://www.evalita.it/campaigns/evalita-2023/) shared framework.

Two different evaluation metrics will be defined according to the task setting:
- For Sub-task 1: the evaluation metric will be based on the Accuracy (as the ratio between all hits and all processed records) obtained by each system in the test set. A second metric will be made also available, in order to grade the errors with respect to the gold results. 
- For Sub-task 2: the evaluation metric will be based on a standard correlation coefficient (Pearson and/or Spearman) between the participants' scores and test set scores.

The baseline for both tasks will be computed by employing the one-hot vectors representation:
- For Sub-task 1: the vector will be extracted from each sentence si in the input prompt P = {s1,s2, ..., sn} and another vector will be created for the target sentence t. The distance between P and the target sentence t, D(P, t) will be computed as the average distance between each pair involving one item si and t based on a distance metric Dist (e.g., Hamming distance, Jaccard, or a Edit distance):

$$\text{D}(P,t) = \frac{1}{n} \left(\sum_{i=0}^{n} \text{Dist} \langle s_i,t \rangle \right)\,$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To decide whether the target sentence t is coherent with the paragraph P we will first compute the median value across the whole training dataset, and then we will use this as a threshold: all the occurrences with a value above the median will be considered coherent, incoherent otherwise.

- For Sub-task 2: the vector will be extracted from each sentence si in the input prompt P = {s1,s2, ..., sn}; that is the following vectors set will be computed:

$${\vec{v_1} \leftarrow s_{1}, \, \vec{v_{2}} \leftarrow s_{2}, \, \dots, \, \vec{v_{n}} \leftarrow s_{n}\}\,.$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The proximity between each two vectors ⟨vx, vx+1⟩ ∈ V will then be computed through a distance metric Dist(s1,s2) (e.g. Jaccard), thereby resulting in (n − 1) distance scores, grasping the degree of semantic overlap between each two neighbouring sentences. In order to compute the coherence score for the paragraph P score(P), we will average the scores featuring each pair of adjacent sentences. The value will then be compared with the human rating with correlation indices:

$$\text{corr}(P) = \frac{1}{n-1} \sum_{i = 1}^{n-1} Dist(v_i, v_{i+1}) \,,$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where corr indicates the Pearson or Spearman correlation index.

## Prerequisites
#### 1 - Install Requirements
```
conda env create -f environment.yml
```
NB. It requires [Anaconda](https://www.anaconda.com/distribution/)

#### 2 - Download data
Download the data from the [DisCoTEX](https://sites.google.com/view/discotex/data) website.
Datasets for all the sub-tasks are released as tab-separated text files. 
For the first sub-task (Last sentence classification) we kept separated data from the two sources (i.e. Wikipedia and TED). Both versions present the following structure:
- ID: a simple identifier for the entry;
- PROMPT: a small snippet of text (3 sentences on average);
- TARGET: the sentence for which participants are asked to assess if it is coherent with the PROMPT (i.e. it is the next sentence after the PROMPT);
- CLASS: the class to be predicted. 1 stands for the positive class (i.e. the TARGET follows the PROMPT), 0 for the negative one (i.e. the TARGET does not follow the PROMPT).

For the second sub-task (Human score prediction) we mixed data from the two sources and we release a single dataset with the followng structure:
- ID: a simple identifier for the entry;
- TEXT: a small snippet of text (4 sentences on average), to be evaluated;
- MEAN: the coherence score of the TEXT to be predicted, based on the mean of the human judgements collected;
- STDEV: standard deviation of the coherence score.

## Execution
#### Download and run the source code
After downloading the code, in order to run the baseline for the first sub-task run the following command:
```
python run_baseline_t1.py \
 --train_file <path to the file containing the training set> \
 --test_file <path to the file containing the test set>
```
where the `--train_file` argument refers to the tab-separated file containing the training set and the `--test_file` refers to the file containing the test set.

In order to run the baseline for the second sub-task run the following command:
```
python run_baseline_t2.py \
 --train_file <path to the file containing the training set> \
 --test_file <path to the file containing the test set>
```
where the `--train_file` argument refers to the tab-separated file containing the training set and the `--test_file` refers to the file containing the test set.