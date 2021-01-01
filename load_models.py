
# loading models
def load(model_path="./models"):
    models = []
    models.append(keras.models.load_model(model_path + "/handwriting.model"))
    models.append(keras.models.load_model(model_path + "/model_3word.h5"))
    models.append(keras.models.load_model(model_path + "/model_4word.h5"))
    models.append(keras.models.load_model(model_path + "/model_5word.h5"))
    models.append(keras.models.load_model(model_path + "/model_6word.h5"))
    models.append(keras.models.load_model(model_path + "/model_7word.h5"))
    models.append(keras.models.load_model(model_path + "/model_8word.h5"))
    return models
