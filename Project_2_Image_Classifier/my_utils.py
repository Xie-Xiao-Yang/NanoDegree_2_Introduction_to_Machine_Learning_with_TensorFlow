import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image


def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 255.0
    return image.numpy()

def predict(image_path, model_name, top_k = 5):
    
    model = tf.keras.models.load_model('./'+model_name, custom_objects={'KerasLayer':hub.KerasLayer})
    
    im = Image.open(image_path)
    im_numpy = np.asarray(im)
    im_numpy = process_image(im_numpy)
    im_numpy = np.expand_dims(a=im_numpy, axis=0)
    ps = model.predict(im_numpy)
    
    classes = ps.argsort()[0][-top_k:][::-1]
    probs = ps[0][ps.argsort()[0][-top_k:][::-1]]
    
    return (probs, classes)