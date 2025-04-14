import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Example training data (replace with your actual data)
X_train_yield = np.random.rand(100, 10)  # 100 samples, 10 features
y_train_yield = np.random.rand(100)

X_train_classifier = np.random.rand(100, 10)
y_train_classifier = np.random.randint(2, size=100)

X_train_fertilizer = np.random.rand(100, 10)
y_train_fertilizer = np.random.randint(2, size=100)

X_train_cr = np.random.rand(100, 10)
y_train_cr = np.random.randint(2, size=100)

# Re-train and save the DecisionTreeRegressor model for yield_rf.pkl
forest = DecisionTreeRegressor()
forest.fit(X_train_yield, y_train_yield)
with open('models/yield_rf.pkl', 'wb') as file:
    pickle.dump(forest, file)

# Re-train and save the RandomForestClassifier model for classifier.pkl
classifier = RandomForestClassifier()
classifier.fit(X_train_classifier, y_train_classifier)
with open('models/classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Re-train and save the RandomForestClassifier model for fertilizer.pkl
fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(X_train_fertilizer, y_train_fertilizer)
with open('models/fertilizer.pkl', 'wb') as file:
    pickle.dump(fertilizer_model, file)

# Re-train and save the RandomForestClassifier model for RandomForest.pkl
cr_model = RandomForestClassifier()
cr_model.fit(X_train_cr, y_train_cr)
with open('models/RandomForest.pkl', 'wb') as file:
    pickle.dump(cr_model, file)

print("All models have been re-trained and saved successfully.")