!pip install wittgenstein pandas scikit-learn
import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("playtennis.csv")


if df['PlayTennis'].dtype == object:
    df['PlayTennis'] = df['PlayTennis'].map({'Yes': True, 'No': False})


train, test = train_test_split(df, test_size=0.2, random_state=42)


ripper_clf = lw.RIPPER(random_state=42, verbosity=0)
ripper_clf.fit(train, class_feat='PlayTennis', pos_class=True)


print("Learned RIPPER Rules:")
print(ripper_clf.ruleset_)


X_test = test.drop('PlayTennis', axis=1)
y_test = test['PlayTennis']


predictions = ['Yes' if p else 'No' for p in ripper_clf.predict(X_test)]
y_test_labels = ['Yes' if y else 'No' for y in y_test]


accuracy = accuracy_score(y_test_labels, predictions)
print(f"Accuracy on test set: {accuracy:.2f}")


print("\nDetailed Rule Set Analysis:")
ripper_clf.out_model()


