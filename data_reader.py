import pandas as pd


class DataReader:

    def __init__(self, directory, train_file, test_file):
        self.directory = directory
        self.train_file = train_file
        self.test_file = test_file

    def obtain_data(self):

        # Obtain the data from the train.csv file
        train = pd.read_csv(self.directory + '/' + self.train_file)

        # Drop the name, passenger id, cabin and ticket columns, since they are not relevant for training
        train = train.drop(['Name'], axis=1)
        train = train.drop(['PassengerId'], axis=1)
        train = train.drop(['Cabin'], axis=1)
        train = train.drop(['Ticket'], axis=1)

        # Convert the sex and embarked columns in to integers
        train['Sex'] = train['Sex'].astype('category').cat.codes
        train['Embarked'] = train['Embarked'].astype('category').cat.codes

        # Give null ages the average age
        age_average = train['Age'].dropna().mean()
        train['Age'] = train['Age'].fillna(age_average)

        # Give null fares the average fare
        fare_average = train['Fare'].dropna().mean()
        train['Fare'] = train['Fare'].fillna(fare_average)

        # Separate x and y in the data set
        x = train.drop(['Survived'], axis=1)
        y = train['Survived']

        # Apply the same for the test.csv set (validation data)
        x_val = pd.read_csv(self.directory + '/' + self.test_file)
        x_val = x_val.drop(['Name'], axis=1)

        # Save passenger id for the output file
        ids = x_val['PassengerId']

        x_val = x_val.drop(['PassengerId'], axis=1)
        x_val = x_val.drop(['Cabin'], axis=1)
        x_val = x_val.drop(['Ticket'], axis=1)
        x_val['Sex'] = x_val['Sex'].astype('category').cat.codes
        x_val['Embarked'] = x_val['Embarked'].astype('category').cat.codes
        age_average = x_val['Age'].dropna().mean()
        x_val['Age'] = x_val['Age'].fillna(age_average)
        fare_average = x_val['Fare'].dropna().mean()
        x_val['Fare'] = x_val['Fare'].fillna(fare_average)

        return x, y, x_val, ids
