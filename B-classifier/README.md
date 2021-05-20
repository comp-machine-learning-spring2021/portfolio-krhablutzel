# Predicting NC's Existing Seasons
### Supervised Learning with Assorted Classifiers

When I was thinking about North Carolina's seasons for the [clustering project](../clustering/README.md), I started wondering how well our traditional seasonal boundaries fit North Carolina's weather. Seasons officially start and end with solstices and equinoxes, but the weather rarely seems to shift between seasons anywhere near these dates. In my part of NC, summer seems to last from May until early October, and winter only visits briefly between January and February—far off from the official seasonal boundaries in late March, June, September, and December. Thus, in this project, I decided to explore two questions: *What classifier model is best for predicting the season of a day based on its weather characteristics? How accurate can this model get?* 

In this notebook, I go through the test, train, and validation stages of building a classifier model. I use 10-fold cross validation to test random forest, decision tree, and k-nearest neighbors classifiers, and I find that a decision tree classifier is the best model for predicting the season of a day of weather in NC based on weather characteristics such as temperature, pressure, and precipitation. Then, I build this decision tree classifier and validate the model on the remaining 10% of the data, which I had previously withheld from testing and training, achieving a validation error of 12.4%. Considering that North Carolina's weather doesn't feel like it fits the official seasons, this seems to be a reasonably accurate model.

See `classifier.ipynb` for a complete walkthrough of this project. This notebook is a complement to the [clustering project](../clustering/README.md), where I throw away the traditional seasonal boundaries and cluster NC's weather into a new set of seasons.

#### References Consulted
- [np.random.choice documentation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
- [StackOverflow elements not in index](https://stackoverflow.com/questions/27824075/accessing-numpy-array-elements-not-in-a-given-index-list)
- [StackOverflow shuffle dataframe rows](https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows)
- [pandas concat DataFrames helpfile](https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html)