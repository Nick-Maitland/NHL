import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data1 = pd.read_csv('/Users/nicolasmaitland/Library/CloudStorage/OneDrive-DurhamCollege/Intro to NN/Assignment2/nhl_data/game_teams_stats.csv')
data2 = pd.read_csv('/Users/nicolasmaitland/Library/CloudStorage/OneDrive-DurhamCollege/Intro to NN/Assignment2/nhl_data/game_skater_stats.csv')

# Merge the two datasets based on a common column 'game_id'
merged_data = pd.merge(data1, data2, on='game_id')

# Print the available columns in the merged DataFrame
#print(merged_data.columns)

# Print the first 10 rows of the merged DataFrame
#print(merged_data.head(10))

# Prepare the features and labels
features = ['goals_x', 'assists', 'shots_x', 'hits_x', 'goals_y', 'shots_y', 'powerPlayGoals_x']
X = merged_data[features].fillna(0)
y = merged_data['won'].astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the ANN model
model = Sequential([
    Dense(X_train.shape[1], activation='relu', input_shape=(X_train.shape[1],)),
    Dense(10, activation='relu'),
    Dense(5, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')