# Multi-lingual Machine Translation

This project was designed to take two datasets french-english and russian-english, and attempt a zero-shot translation between French and Russian.
Two bi-lingual models were created as well, French-English and Russian-English, in order to establish a baseline to compare the multilingual model to.

# Installation
To work with the repository you need to install required packages and this package. Edit (-e) mode is perferred if you want to change the code.

```
pip install -r requirements.txt
```

## Training the Model

In order to train this model, you will first want to navigate to the `config.toml` file and modify the hyper-parameters there to your desiered values. After that, you will simply need to execute 
``` 
python3 main.py 
```
to begin training the model

