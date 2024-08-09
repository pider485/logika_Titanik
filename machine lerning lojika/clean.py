import pandas as pd

df = pd.read_csv("titanic.csv")
df.info()

# print(df.groupby('Sex')['Survived'].mean())
# print(df.pivot_table(index='Survived', columns="Pclass", values="Age", aggfunc='mean'))

# print(df.groupby("Embarked")['Survived'].mean())

# print(df.groupby("SibSp")['Survived'].mean())

# print(df.groupby("Parch")['Survived'].mean())


age1 = df[df['Pclass']== 1]["Age"].median()
age2 = df[df['Pclass']== 2]["Age"].median()
age3 = df[df['Pclass']== 3]["Age"].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass']== 1 :
            return age1
        if row['Pclass']== 2 :
            return age2
        else:
            return age3
    else:
        return row["Age"]

df['Age'] = df.apply(fill_age, axis=1)
df['Embarked'].fillna('S', inplace=True)

def fill_sex(data):
    if data == "male":
        return 1
    return 0

df['Sex']= df['Sex'].apply(fill_sex)


df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])



def is_alone(data):
    if data["SibSp"]+ data['Parch'] == 0:
        return 1
    return 0
df.drop(['Name','Ticket','Cabin','Embarked'], axis=1 , inplace=True)

df['Alone'] = df.apply(is_alone, axis= 1)

df.info()

df.to_csv("Clean_titanik.csv", index=False)