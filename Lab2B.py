import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("mushrooms.csv")
df['class'] = df['class'].map({'e': False, 'p': True})


train, test = train_test_split(df, test_size=0.3, random_state=42)


ripper_clf = lw.RIPPER(random_state=42, verbosity=0)
ripper_clf.fit(train, class_feat='class', pos_class=True)


print("Learned RIPPER Rules:")
print(ripper_clf.ruleset_)


X_test = test.drop('class', axis=1)
y_test = test['class']
predictions = ripper_clf.predict(X_test)


accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy on test set: {accuracy:.4f}")


print("\nDetailed Rule Set Analysis:")
ripper_clf.out_model()


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=["Edible", "Poisonous"]))
