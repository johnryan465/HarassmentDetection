# HH_TensorFlow
```
sudo pip install tensorflow
sudo pip install keras
sudo pip install numpy
sudo pip install panadas
sudo pip install h5py
```

To train a model run ```python learning.py``` with the correct arguements passed.
```
usage: learning.py [-h] -n NAME [-e EPOCHS] [--maxlen MAXLEN]
                   [--model_size MODEL_SIZE] [--ratio RATIO]
                   [--batch_size BATCH_SIZE] [--dropout DROPOUT]
                   [--alphabet ALPHABET]

optional arguments:
  -h, --help            show this help message and exit
  -n NAME, --name NAME  Model Name
  -e EPOCHS, --epochs EPOCHS
                        Number of Epochs
  --maxlen MAXLEN       Length of the sequences
  --model_size MODEL_SIZE
                        Amount of LSTM units
  --ratio RATIO         Ratio of harassing datapoint error
  --batch_size BATCH_SIZE
                        Batch size used during training
  --dropout DROPOUT     Dropout used in training
  --alphabet ALPHABET   Alphabet used for encoding the text

```
If you wish to change the model structure, edit the build_model function in utils.py

val.py loads a trained model and iterates through the dataset and prints out examples that the model predicted incorrectly, you may notice that some of the items that are printed have questionable labels.

 ```python val.py -n [model_name]```

To change the dataset used to train the model, you must edit the process_data function or create your own to perform the task.
