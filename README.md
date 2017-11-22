theres one notebook - train_test_devision - impossible to run due to the lack of clicks_train (was done since the original table was too heavy).

all of the tables are already in the zips, but it’s also possible to rebuild them using the initialization notebook.

both timezone.py and ctr_features.py are run through initialization notebook but there are also working notebooks for both of them inside the notebooks folder.

categories_topics_dicts are for building the correlation dictionaries for topics and categories.. you can set the parameters for how to cut the tables for making these.
the dictionaries are saved into the dicts folder.

in order to change a used dictionary the dictionarys’ name has to be changed inside the logistic notebook.

in the logistic notebook everything runs well and built well until the algorithm running.
the grid search is not yet tested and there is a cell there for checking all 2^8 combinations of different features but it’s quite messy and can be done more beautifuly with a function, not a must..
