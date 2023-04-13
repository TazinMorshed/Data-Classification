
# Classification problem 

In this projects, I have used 4 machine learning model to perform classification on 3 datasets which was taken from [ scikit-learn website.](https://scikit-learn.org/stable/datasets/toy_dataset.html). After classification, I have used **stacking ensemble learning** and compared the accuracy. 




## Datasets

In this projects I have used the following datasets : 

- Iris plant dataset.
- Wine recognition dataset.
- Breast cancer wisconsin (diagnostic) dataset.


## Models Used
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine


## Approach
First load the dataset using `load()` function from scikit-learn. Then, split the dataset into training and testing sets (80:20) using `train_test_split()` function. Next, define 4 machine learning models (Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine) using their corresponding classes from scikit-learn. Train each model on the training set using the `fit()` method. 

I have apply cross-validation score on the training dataset using `cross_val_score()` function from scikit-learn. Then, iterate over the models and print the accuracy of each model using the mean of the cross-validation scores. After that, define a meta-model for stacking based on the Decision Tree Classifier. Train the meta-model on the meta features. Then, make predictions on the testing set using the stacked model. Finally compute the accuracy of the stacked model using the `accuracy_score()`.


## Stacking-based ensemble learning 

In stacking-based ensemble learning, I use the predictions made by multiple base models to train a meta-model that makes the final prediction. The meta-model is trained on the meta features instead of the original features. The meta-model learns how to combine the predictions of the base models to make the final prediction. In my code, I have defined Decision Tree Classifier as the meta-model for stacking. This is because decision trees are simple yet powerful models that can handle both categorical and numerical data. They are also good at capturing non-linear relationships between the features and the target variable. 

## Results 

####  Comparison of accuracy on test data
| Models | Iris Data     | Wine Data   | Cancer Data     |
| :-------- | :------- | :------------------------- | | :------------------------- |
| Decision Tree | 96.67% | 88.57% |  92.92%|
| Random Forest | 100.00% |91.79%| 93.83% |
| K-Nearest Neighbors | 93.33% | 80.71%| 95.65%|
| Support Vector Machine | 96.67% | 94.29% |96.44% |
| Stacking Ensemble Learning | **100.00%** | **97.22%**|**98.25%** |


## Result Analysis

Our models gave good accuracy score, however `100% accuracy` on  Stacking Ensemble Learning might make it seem like a case of overfitting. However this is not the case as the accuracy is given by prediction of the model given completely unseen data.  


One possible reason for getting 100% accuracy using the stacking-based ensemble learning approach is that the dataset (iris-dataset) is relatively small and simple.


## 🚀 Reach out

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tazin-morshed-b441a6237/)