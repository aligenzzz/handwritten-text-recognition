from keras.models import load_model
import random
import matplotlib.pyplot as plt

from constants import ENGLISH_MODEL, DESIRED_SIZE
from data_formatter import dataset_to_sample, get_encoded, get_test_images
from segmentation import word_segmentation
from spellchecker import SpellChecker

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
