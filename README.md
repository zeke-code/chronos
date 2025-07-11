# Chronos

**Chronos** is a small ML model to make forecast predictions about the energy load in a certain country.

The example provided in this repo trains the model on Italy's Energy Consumption dataset, provided by [Terna](https://dati.terna.it/en/). You can find the raw data under the
`raw/` folder. The data is then processed and a lot of features are added to make the model more accurate in its predictions.

## How to run

To run a complete training and evaluation cycle, follow these steps:

1. Download **[Anaconda](https://www.anaconda.com/)** and clone this repo
2. With Anaconda, run the following command and install all the required dependencies.

```bash
conda env create -f environment.yml
```

3. Run the following commands once dependencies are installed

```bash
python src/data_preprocessing.py
python src/train.py
python src/evaluate.py
```

The following commands will preprocess our data, train the model with 50 epochs (you can modify `config.json` to experiment with hyperparameters) and evaluate the model
on data it has never seen.

## Tensorboard support

You can have a graphical representation of what's happening during training through Tensorboard. Run the following command and open `http://localhost:6006/`
on your browser to consult Tensorboard graphs.

```bash
tensorboard logdir=results
```
