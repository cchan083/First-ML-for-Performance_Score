# Performance Score ML

## Dataset from kaggle: [Music_dataset](https://www.kaggle.com/datasets/ziya07/music-education-performance-data)

The features I picked:

```features = ['Accuracy', 'Rhythm', 'Pitch_Accuracy', 'Behavioral_Patterns', 'Focus_Time', "Engagement_Score"]```

I used a decision tree regressor

### Split the training and test data in a 25:75 ratio with the default value

```train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)```

### Training the model with .fit()
```performance_model.fit(train_X, train_y)```

## Calculating the Mean absolute error

```val_mae = mean_absolute_error(predictions, val_y)```