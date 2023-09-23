from sklearn.ensemble import RandomForestClassifier
from data_formatter import dataset_to_training_sample, vectorize_image

model = RandomForestClassifier()

x, y = dataset_to_training_sample()
model.fit(x, y)
print(x[0].shape)


example = [vectorize_image(f'images/{i + 1}.png') for i in range(10)]
print(example[0].shape)

for ex in example:
    predictions = model.predict(ex.reshape(1, -1))
    print(predictions)
