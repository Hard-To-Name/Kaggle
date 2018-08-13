from IPython.display import display # DataFrame Display
import matplotlib.pyplot as plt # Data/Statistics Plot
import numpy as np # Scientific Calculation
import pandas as pd # Data Analysis
from pandas import Series, DataFrame # Data Structure
from sklearn.ensemble import RandomForestRegressor # Training
import sklearn.preprocessing as preprocessing # Scaling
from sklearn import linear_model # Logistic Regression
from sklearn.model_selection import learning_curve # Track Learning Curve

data_train = pd.read_csv("train.csv")

### UNDERSTANDING DATA & CAPTURE FEATURES ###

'''

# display(data_train)
data_train.info()
# print(data_train.describe(include = 'all'))

fig = plt.figure()
fig.set(alpha = 0.2) # Set Chart Color

# Subplots 2x3

# Figure 1
plt.subplot2grid((2, 3), (0, 0))
data_train.Survived.value_counts().plot(kind = 'bar') # Bar Chart
plt.title("Survived")
plt.ylabel("# of People")

# Figure 2
plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind = 'bar') # value_counts(): frequencies of unique values
plt.title("Passenger Rank")
plt.ylabel("# of People")

# Figure 3
plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Survived, data_train.Age)
plt.title("Survived")
plt.ylabel("Age")
plt.grid(b = True, which = 'major', axis = 'y')

# Figure 4
plt.subplot2grid((2, 3), (1, 0))
data_train.Age[data_train.Pclass == 1].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 2].plot(kind = 'kde')
data_train.Age[data_train.Pclass == 3].plot(kind = 'kde')
plt.title("Age Distribution of Different Ranks")
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(("First Class", "Second Class", "Third Class"), loc = 'best')

# Figure 5
plt.subplot2grid((2, 3), (1, 1))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title("# of People of Different Port No.")
plt.ylabel("# of people")

# Figure 6
plt.subplot2grid((2, 3), (1, 2))
data_train.Sex[data_train.Survived == 1].value_counts().plot(kind = 'bar')
plt.title("# of Survived People of Different Sex")
plt.ylabel("# of Survived People")

# Subplots 1x1
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({"Saved": Survived_1, "Unsaved": Survived_0})
df.plot(kind = 'bar', stacked = True) # value_counts(): frequencies of unique values
plt.title("Saved Proportion of Different Passenger Rank")
plt.xlabel("Passager Rank")
plt.ylabel("# of People")

# Subplots 1x1
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({"Male": Survived_m, 'Female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("Saved Proportion of Different Gender")
plt.xlabel("Gender") 
plt.ylabel("# of People")

plt.show()

# SibSp
g = data_train.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print(df)

# Ticket No.
print(data_train.Cabin.value_counts())

'''

### FEATURE ENGINEERING ###

def set_missing_ages(df):
    # Selected Features
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 2 Groups: Known/Unknown Ages
    known_age   = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    
    # Training Data
    X = known_age[:, 1:]
    y = known_age[:, 0]

    # Predict Unknown Ages
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
    rfr.fit(X, y)
    age_prediction = rfr.predict(unknown_age[:, 1:])

    # Fit Missing Ages
    df.loc[(df.Age.isnull()), 'Age'] = age_prediction

    return df, rfr

def set_Cabin_type(string):
    if string == "nan": return 0
    if string[0] == 'T': return 3
    elif string[0] in ['U', 'B', 'D', 'E']: return 2
    else: return 1

data_train, rfr = set_missing_ages(data_train)
# print(data_train)

# Add Feature 'Child', 'Family_Size', 'Cabin_No', 'Title'
df_child = pd.DataFrame({'Child': [int(key <= 12) for key in data_train['Age']]})
df_family_size = pd.DataFrame({'Family_Size': [data_train['SibSp'][key] + data_train['Parch'][key] for key in range(len(data_train.index))]})
df_cabin_no = pd.DataFrame({'Cabin_No':[set_Cabin_type(str(key)) for key in data_train['Cabin']]})

replacement = {
    'Don': 0,
    'Dona': 6,
    'Rev': 0,
    'Jonkheer': 0,
    'Capt': 0,
    'Mr': 1,
    'Dr': 2,
    'Col': 3,
    'Major': 3,
    'Master': 4,
    'Miss': 5,
    'Mrs': 6,
    'Mme': 7,
    'Ms': 7,
    'Mlle': 7,
    'Sir': 7,
    'Lady': 7,
    'the Countess': 7
}

def title_segment(string:str):
    start = end = 0
    for i in range(len(string)):
        if string[i] == ',':
            start = i + 2
        elif string[i] == '.':
            end = i
            break
    return string[start:end]

df_title = pd.DataFrame({'Title': [replacement[title_segment(key)] for key in data_train['Name']]})

# Turn Classifier Type into Quantifier Type
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix = 'Embarked')
dummies_Sex      = pd.get_dummies(data_train['Sex'], prefix = 'Sex')
dummies_Pclass   = pd.get_dummies(data_train['Pclass'], prefix = 'Pclass')

# Modify Training DataFrame (add 'Quantifier' features, remove 'Classifier' features)
df = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass, df_child, df_family_size, df_cabin_no, df_title],  axis = 1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)

# Feature Scaling
scaler = preprocessing.StandardScaler()
age_scale_param  = scaler.fit([[element] for element in df['Age']])
df['Age_scaled'] = scaler.fit_transform([[element] for element in df['Age']], age_scale_param)
fare_scale_param  = scaler.fit([[element] for element in df['Fare']])
df['Fare_scaled'] = scaler.fit_transform([[element] for element in df['Fare']], fare_scale_param)
parch_scale_param  = scaler.fit([[element] for element in df['Parch']])
df['Parch_scaled'] = scaler.fit_transform([[element] for element in df['Parch']], parch_scale_param)
cabin_scale_param  = scaler.fit([[element] for element in df['Cabin_No']])
df['Cabin_scaled'] = scaler.fit_transform([[element] for element in df['Cabin_No']], cabin_scale_param)

'''
fig = plt.figure()
fig.set(alpha = 0.2) # Set Chart Color
plt.subplot2grid((1, 3), (0, 0))
df.Age_scaled.plot(kind = 'kde')
plt.subplot2grid((1, 3), (0, 1))
df.Fare_scaled.plot(kind = 'kde')
plt.subplot2grid((1, 3), (0, 2))
df.Parch_scaled.plot(kind = 'kde')
plt.show()
'''

# Select Useful Features
train_df = df.filter(regex = 'Survived|Age_scaled|SibSp|Parch_scaled|Cabin_scaled|Embarked_S|Sex_.*|Pclass_.*|Child|Family_Size|Title')
train_np = train_df.as_matrix()

# Select all columns except 'Survived' as X
X = train_np[:, 1:]
# Select 'Survivied' column as y
y = train_np[:, 0]

# Train Logistic Regresion Model
clf = linear_model.LogisticRegression(C = 2.0, penalty = 'l2', tol = 1e-6)
clf.fit(X, y)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    # Parameters:
    # estimator - Classifier (e.g. clf)
    # title     - Title of table
    # X         - Input feature，type numpy
    # y         - Input target vector
    # ylim      - (ymin, ymax) -> scale
    # cv        - Cross validation
    # n_jobs    - # of parallel tasks
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("# of Training Samples")
        plt.ylabel("Score")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label= "Training Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label= "CV Score")

        plt.legend(loc="best")

        plt.draw()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, "Learning Curve", X, y)

### Modify test data ###

# Read data
data_test = pd.read_csv("test.csv")

# Column 'Fare': fill null with 0
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

# Column 'Age': fill null with predicted ages
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

# Add Feature 'Child', 'Family_Size', 'Title'
df_child = pd.DataFrame({'Child': [int(key <= 12) for key in data_test['Age']]})
df_family_size = pd.DataFrame({'Family_Size': [data_test['SibSp'][key] + data_test['Parch'][key] for key in range(len(data_test.index))]})
df_cabin_no = pd.DataFrame({'Cabin_No':[set_Cabin_type(str(key)) for key in data_test['Cabin']]})
df_title = pd.DataFrame({'Title': [replacement[title_segment(key)] for key in data_test['Name']]})

# Turn Classifier Type into Quantifier Type
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix = 'Embarked')
dummies_Sex      = pd.get_dummies(data_test['Sex'], prefix = 'Sex')
dummies_Pclass   = pd.get_dummies(data_test['Pclass'], prefix = 'Pclass')

# Modify Testing DataFrame (add 'Quantifier' features, remove 'Classifier' features)
df_test = pd.concat([data_test, dummies_Embarked, dummies_Sex, dummies_Pclass, df_child, df_family_size, df_cabin_no, df_title],  axis = 1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis = 1, inplace = True)
df_test['Age_scaled'] = scaler.fit_transform([[element] for element in df_test['Age']], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform([[element] for element in df_test['Fare']], fare_scale_param)
df_test['Parch_scaled'] = scaler.fit_transform([[element] for element in df_test['Parch']], parch_scale_param)
df_test['Cabin_scaled'] = scaler.fit_transform([[element] for element in df_test['Cabin_No']], cabin_scale_param)

# Select Useful Features
test_df = df_test.filter(regex = 'Age_scaled|SibSp|Parch_scaled|Cabin_scaled|Embarked_S|Sex_.*|Pclass_.*|Child|Family_Size|Title')

predictions = clf.predict(test_df)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv("predictions.csv", index = False)
