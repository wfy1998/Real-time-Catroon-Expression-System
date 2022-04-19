# Catroon-Expression-system

### Dependencies
* [Python](https://www.python.org/) - The code is designed to work with Python 2.7 and Python 3.5 or greater.
* [NumPy](http://www.numpy.org/) -  The randomized routines used for stochastic simulations are implemented using NumPy.
* [TensorFlow](https://www.tensorflow.org/) - The core open source library to help you develop and train ML models. Get started quickly by running Colab notebooks directly in your browser.
* [Pandas](https://pandas.pydata.org/) - pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.




## Runing Collect data
To collect training and test data, run `data_record.py`. Various  parameters are available in the file. The record data will be store into a folder call `data`

```console

optional arguments:
--train                  Record training data or testing data. 1 for training data. 0 for testing data.
--store_method           Choose the what kinds of store method you want to use. 1 for record the facial landmarks data. 0 for facial image
--expression_ID          The data expression ID tag.
--show_landmarks         If you want to show the landmark points in the camera screen.
--relative_coordinates   Take the first landmark point as base point. Calculate the position of the rest of points based on the origin.

```

## Train data

```console

python landmark_train.py
python image_train.py
```

## Real-time tracking

```console

python real_time_tracking.py
