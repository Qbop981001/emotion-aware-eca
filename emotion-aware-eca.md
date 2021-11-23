# emotion-aware-ecpe

This repository show the source code for the submitted paper "Learning Emotion-Aware Contextual Representations for Emotion-Cause Pair Extraction"

In directory ```emotion-aware ecpe``` , there are five folders.

```ECPE-original``` and ```ECPE-reconstructed```  are our datasets. ```ECPE-original``` is the original ECPE benchmark dataset, which are used by all previous ECPE works,  ```ECPE-reconstructed``` is our reconstructed dataset. We build this  dataset by manually checking all documents and merge those that are from the same original news article.

```emotion-model``` and ```cause-model```  are source code of our approach. 

In ```emotion-model``` ,  run

```python train_emotion.py``` for training and saving model checkpoints

```python predict_emotion.py```  for predicting and storing the emotion clauses for each fold 

You can change the variable

```DATASET_TYPE``` (set to 'original' by default) in ```data_processing_emotion.py```  to select the dataset you want to run experiments on,

By default, predicted emotion clauses will be stored in ```../predictions_original```  or ```../predictions_reconstructed``` for the two datasets respectively







