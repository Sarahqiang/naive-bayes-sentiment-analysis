from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Gaussiannb import X_train_tfidf
from Gaussiannb import  X_train_counts
from Gaussiannb import  X_new_tfidf
from Gaussiannb import  X_new_counts
from Gaussiannb import  twenty_train
from Gaussiannb import twenty_test
#change the internal variable according to the test. tHe best probability for test 1 should be 66 percent.
#the best probability for test 2 should be 85
model = LogisticRegression()
model.fit(X_train_tfidf.toarray(),twenty_train.target)
y_prob = model.predict(X_new_tfidf.toarray())
accuracy_score(y_prob,twenty_test.target)