from keras.models import load_model
import random
import matplotlib.pyplot as plt

from constants import ENGLISH_MODEL, DESIRED_SIZE
from data_formatter import dataset_to_sample, get_encoded, get_test_images
from segmentation import word_segmentation
from spellchecker import SpellChecker

# for Raspberry Pi
# from tensorflow.lite.python.interpreter import Interpreter
# # from tflite_runtime.interpreter import Interpreter

# tflite_model_path = './models/english_model.tflite'

# english_model = Interpreter(model_path=tflite_model_path)
# english_model.allocate_tensors()

# input_tensor = english_model.get_input_details()[0]
# output_tensor = english_model.get_output_details()[0]

english = SpellChecker()

english_model = load_model(ENGLISH_MODEL)

def test_prediction():
    x, y, encoded = dataset_to_sample()
    x = x.reshape(-1, DESIRED_SIZE[0], DESIRED_SIZE[1], 1)

    predictions = english_model.predict(x)

    displayed_samples = 50
    random_indexes = random.sample(range(len(x)), displayed_samples)

    fig = plt.figure(figsize=(12, 8))

    for i, index in enumerate(random_indexes):
        sf = fig.add_subplot(5, 10, i + 1)
        sf.imshow(x[index], cmap='gray')

        predicted_label = encoded[predictions[index].argmax()]
        true_label = encoded[predictions[index].argmax()]

        sf.set_title(f'Predicted: {predicted_label}\nReality: {true_label}')
        sf.axis("off")

    plt.tight_layout()
    plt.show()


def word_prediction(characters):
    if len(characters) == 0:
        return ''

    encoded = get_encoded()
    characters = get_test_images(characters)
    predictions = english_model.predict(characters)

    predictions = [encoded[prediction.argmax()].lower() for prediction in predictions]
    return predictions


# for Raspberry Pi
# encoded = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 
# 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}

# def word_prediction(characters):
#     if len(characters) == 0:
#         return ''

#     characters = get_test_images(characters)
    
#     predictions = []
#     for character in characters:
#         english_model.set_tensor(input_tensor['index'], character)
#         english_model.invoke()
#         predictions.append(english_model.get_tensor(output_tensor['index']))


#     predictions = [encoded[prediction.argmax()].lower() for prediction in predictions]
#     return predictions


def predict(file_name):
    words = word_segmentation(image_path=file_name)
    text = ''
    for word in words:
        word = word_prediction(word)

        word = ''.join(word)

        result = english.correction(word)
        if result is None:
            text += word + ' '
        else:
            text += result + ' '

    return text


if __name__ == '__main__':
    # test_prediction()

    result_text = predict('images/lines3.png')
    print(result_text)
