# testing_hyperband
A little repository with testing purpose to learn how works Hyperband algorithm from keras_tuner

# Description
In this repository, you will find how to:
- Load images (classified by directories) using the `tf.keras.utils.image_dataset_from_directory()` tools
- Create a special Model (HyperModel) with hyperparameters capable of fine-tuning
- Use the [Hyperband](https://keras.io/api/keras_tuner/tuners/hyperband/) from keras_tuner 

# Requirements
- [Poetry](https://python-poetry.org/)
- [Graphviz](https://graphviz.org/)

# Instructions
- Download the [Dataset](https://www.kaggle.com/datasets/l3llff/flowers) from [Kaggle](https://www.kaggle.com/)
- In the same directory with the *pyproject.toml*, run the next command:
````bash
poetry install
````
- The main code is in *testing_kerastuner/main.py*, feel free to change vars to testing, 
you will also need to change the dataset path
- Now it's time to execute:
````bash
poetry run python testing_kerastuner/main.py
````
