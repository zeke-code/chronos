# Chronos

**Chronos** is a small ML model to make forecast predictions about the energy load in a certain country.

The example provided in this repo trains the model on Italy's Energy Consumption dataset, provided by [Terna](https://dati.terna.it/en/download-center). You can find the raw data under the `raw/` folder. The data is then processed and a lot of features are added to make the model more accurate in its predictions.

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

## Experiments

I ran some experiments on my model, experimenting with hyperparameters and the two different architectures I've implemented (GRU and LSTM).

| Experiment Name      | Model Type | Key Parameters Changed       | Final Val Loss | Test MAE (MW) | Test RMSE (MW) |
| :------------------- | :--------- | :--------------------------- | :------------- | :------------ | :------------- |
| **1. Baseline LSTM** | `lstm`     | none, target value disabled  | _0.000134_     | _3219.23 MW_  | _5006.08 MW_   |
| **2. GRU Model**     | `gru`      | Switched model type to `gru` | _0.000112_     | _3188.73 MW_  | _4986.66 MW_   |
| **3. Deeper LSTM**   | `lstm`     | `num_layers=4`               | _0.000168_     | _3244.43 MW_  | _5077.91 MW_   |
| **4. Wider LSTM**    | `lstm`     | `hidden_size=512`            | _0.000613_     | _3277.23 MW_  | _5006.49 MW_   |

### Baseline LSTM

This model performed pretty badly. It made me doubt my code and my architecture and training loop. I've decided to modify the other hyperparameters as well to see if I was the problem. I did not set a target validation loss for this experiment, maybe that was the problem. The model probably ended up overfitting on the training data.

### GRU Model

This model performed slightly better than baseline LSTM and took less time training due to its architecture. Still, it's not extremely accurate.

## Deeper LSTM

This model did roughly like baseline LSTM, taking much more time to

### Wider LSTM

This model takes much more time for training, but shows potentially the best result long term. Training stopped at 5 epochs due to a lack of time, but performance was ok.
