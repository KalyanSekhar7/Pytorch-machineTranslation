from configparser import ConfigParser

config = ConfigParser()

config["TRAINING"] = {

    "num_epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 64

}

config["MODEL"] = {
    "load_model": False
}

config["ENCODER"] = {
    "encoder_embedding_size": 300,
    "hidden_size": 1024,
    "num_layers": 2,
    "encoder_dropout": 0.5
}

config["DECODER"] = {
    "decoder_embedding_size": 300,
    "hidden_size": 1024,
    "num_layers": 2,
    "decoder_dropout": 0.5
}


with open("lstm_config.ini","w") as f:
    config.write(f)