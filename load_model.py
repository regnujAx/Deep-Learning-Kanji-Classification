
from tensorflow import keras

def load_model(model_path):
    # load a model
    loaded_model = keras.models.load_model(model_path)
    return loaded_model