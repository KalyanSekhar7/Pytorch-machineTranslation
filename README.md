# Pytorch-machineTranslation ( English -> French)


![image](https://user-images.githubusercontent.com/98607718/223472387-6c2a5d7b-afc3-4ed1-85a2-6c1a2ad23dff.png)


This is a Machine translation program with 2 main files


1. Simple LSTM model with ( Encoder - Decoder - Sequence to sequence) models
2. LSTM model with Attention ( where the states of each output is as the input in the decoder state)

To run the Simple LSTM machine translator just type

```python machine_translation_lstm.py ```

For Attention model use 

```python translation_attention.py ```

## Changing the config:
The configuration related to 
1. Embedding layers
2. Hidden layers
3. Save model
4. Number of features
6. Number of epoch
5. Model learning rate

is stored in the file   ``` lstm_config.ini```


## Dataset 

The dataset used in this project is  taken from https://www.manythings.org/anki/

TODO: 

  1.Add the model in Fast API router

  2. Create a streamlit app 
