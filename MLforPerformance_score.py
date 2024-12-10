# This machine learning model predicts the Performance score of a student with features 
# ['Accuracy', 'Rhythm', 'Pitch_Accuracy', 'Behavioral_Patterns', 'Focus_Time', "Engagement_Score"]


import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = 'music_education_dataset.csv'

performance_data = pd.read_csv(data)
print(performance_data.head())

features = ['Accuracy', 'Rhythm', 'Pitch_Accuracy', 'Behavioral_Patterns', 'Focus_Time', "Engagement_Score"] #Features affecting performance score
X = performance_data[features]
y = performance_data.Performance_Score
#Splitting test and training data for the dataset
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#model specification
# 500 for max_leaf_nodes
performance_model = DecisionTreeRegressor(max_leaf_nodes=500, random_state=1)
performance_model.fit(train_X, train_y)
predictions = performance_model.predict(val_X)
print(predictions)
val_mae = mean_absolute_error(predictions, val_y)
print(val_mae)