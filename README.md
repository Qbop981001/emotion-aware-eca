# emotion-aware-ecpe
## Implementation Details

This repository show the source code for the submitted paper "Learning Emotion-Aware Contextual Representations for Emotion-Cause Pair Extraction"

In directory ```emotion-aware ecpe``` , there are six folders.

```ECPE-original``` and ```ECPE-reconstructed```  are our datasets. ```ECPE-original``` is the original ECPE benchmark dataset, used by all previous ECPE works,  ```ECPE-reconstructed``` is our reconstructed dataset. We build this  dataset by manually checking all documents and merge those that are from the same original news article.

```emotion-model``` and ```cause-model```  are source code of our approach. 

- In directory ```emotion-model``` ,  run

  ```python train_emotion.py``` for training and evaluating the emotion model

  ```python predict_emotion.py```  for predicting and storing the emotion clauses for each fold 

  You can change the variable

  ```DATASET_TYPE``` in ```data_processing_emotion.py``` (set to 'original' by default) to select the dataset you want to run experiments on,

  By default, predicted emotion clauses will be stored in ```predictions_original```  or ```predictions_reconstructed``` for the two datasets respectively.

  On the other hand, we have also provided the predicted emotion clause for the two datasets by our emotion model, stored in ```predictions_original```  and ```predictions_reconstructed``` , if you don't want to generate the emotion clauses by runnning emotion-model yourself, you can use them directly.

- In directory ```cause-model``` ,  run

  ```python train_emotionaltext.py``` or   ```python train_untypedmarker.py``` or ```python train_typedmarker.py``` for training and evaluating the cause model with different emotion-fusing strategies **EmotionalText** **UntypedMarker** **TypedMarker** respectively. 
  
  For training on the reconstructed dataset, change the variable  ```DATASET_TYPE``` to 'reconstructed' in ```data_processing_emotionaltext.py``` or ```data_processing_untypedmarker.py```. When evaluating on the reconstructed dataset, results on multiple emotion-cause pair extraction will also be printed, and in ```train_typedmarker.py```, we report the upper bound of ECPE.

## Implementation for another data split

In addition to the 10-fold cross-validation data split employed by most previous methods (following https://github.com/Determined22/Rank-Emotion-Cause ), there is another data split based on randomly sampling train/validation/test sets with 8:1:1 proportion 20 times (following https://github.com/HLT-HITSZ/TransECPE). 

We also conduct experiments and report comparative results on this data split in our paper. The code is located in  ```emotion-aware-ecpe-split20```, and the implementation procedure is the same as mentioned above.

## Multi-label learning scheme in the previous version

In our previous version, we employed a different model architecture. Actually, we also tried to implement a multi-label learning scheme in previous experiments while the results were not satisfying, thus, we reported the negative results in the appendix of our previous version. 

However, we found that the unsatisfying results were due to **wrong code implementation** of the multi-label learning scheme (we passed wrong parameter to function `BertModel.from_pretrained()`, leading to the fact we did not download the right bert model). After we modified the code, we found that the results have become reasonable as reported in the current version.  



