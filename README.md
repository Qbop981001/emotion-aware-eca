# emotion-aware-eca
This repository show the source code for the submitted paper "Learning Emotion-Aware Contextual Representations for Emotion Cause Analysis" 

To train the emotion model, run train_emotion.py, 

checkpoints will be stored in "checkpoint_emotion" 

then run predict_emotion.py to predict emotion results and lexicon-revised emotion results, results will be stored in "predictions" 

OR you can directly use the predicted emotion results by our model we provided in the same directory.  

To train and evaluate the cause extraction model based on qa/untyped/typed_markers 

run train_qa.py/train_untyped_marker.py/train_typed_marker.py

By default, the model will use the lexicon-calibrated emotions, if you want to reproduce the results with the original results of emotion extraction you can set the value LEXICON in the data_processing_\*.py file to FALSE

As we are clearing up our code, we will add environment requirements and provide better code for easier training. 

Requirements:\
pytorch\
transformers\
numpy\
sklearn\
pandas
