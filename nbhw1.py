import pandas as pd
from sklearn.naive_bayes import MultinomialNB

def main():
    d_train = read_data('spam-training.txt')
    l_train = read_data('train-labels.txt', True)
    d_test = read_data('spam-testing.txt')
    l_test = read_data('test-labels.txt', True)

    mnb = MultinomialNB()
    train_pred = mnb.fit(d_train, l_train).predict(d_train)
    print("Number of mislabeled train points out of total %d points : %d" % (d_train.shape[0], (l_train != train_pred).sum()))

    test_pred = mnb.fit(d_test, l_test).predict(d_test)
    print("Number of mislabeled test points out of total %d points : %d" % (d_test.shape[0], (l_test != test_pred).sum()))

def read_data(name, labels = False):
    tmp = pd.read_csv(name, header=None)

    if labels:
        return tmp.values.reshape(tmp.shape[0])
    else:
        return tmp.values

main()