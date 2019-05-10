# ===================================================================================================
# Author: Mohammadreza Hajy Heydary
# ===================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ===================================================================================================
SEED = 97
HARD_START = False
NUM_FOLDS = 10


# ===================================================================================================
def preprocess(db):
    pca = PCA(n_components=16,svd_solver ="full", random_state=SEED)
    X = pca.fit_transform(db.iloc[:, :-1])

    # plt.figure(figsize=(8.5, 4), dpi=1200)
    # plt.bar(np.arange(0, len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    # plt.show()

    # accumulator = 0
    # var_explained = []
    # for i in pca.explained_variance_ratio_[0:51]:
    #     accumulator += i
    #     var_explained.append(accumulator)
    #
    # print(var_explained)
    # plt.figure(dpi=1200)
    # plt.plot(var_explained, '.-')
    # plt.show()

    X = pd.DataFrame(X)
    X['l'] = db.iloc[:, -1].values

    X.to_csv('preprocessed.csv', index=False, header=False)
    return X


# ===================================================================================================
def selectTopFeatures(X, Y, Folds, numFeatures):
    scores = []
    for train_index, test_index in Folds.split(X, Y):
        select_feature = SelectKBest(k=numFeatures).fit(X[train_index, :], Y[train_index])
        scores.append(select_feature.scores_)

    scores = pd.DataFrame(scores).mean().values
    sorted_scores = np.sort(scores)[::-1]
    idx = []
    for s_elem in sorted_scores:
        for indx, e in enumerate(scores):
            if e == s_elem:
                idx.append(indx)
                break

        if len(idx) == numFeatures:
            break

    # plt.figure(dpi=1200)
    # h = pd.DataFrame(scores).mean().values
    # plt.bar(np.arange(0, len(h)), h)
    # plt.ylabel('ANOVA F-value ')
    # plt.xlabel('Feature Number')
    # plt.show()
    return X[:, idx]


# ===================================================================================================
def KNN(X, Y, Folds):
    print("KNN training started...")
    n_neighbors = [1, 2, 4, 6, 8, 10, 16, 32]
    knn_results = []
    for n in n_neighbors:
        train_res = []
        test_res = []
        for train_index, test_index in Folds.split(X, Y):
            knn = KNeighborsClassifier(n_neighbors=n, metric='minkowski', n_jobs=-1)
            knn.fit(X[train_index, :], Y[train_index])
            train_res.append(accuracy_score(Y[train_index], knn.predict(X[train_index, :])))
            test_res.append(accuracy_score(Y[test_index], knn.predict(X[test_index, :])))

        print(np.mean(train_res), np.mean(test_res))
        knn_results.append([np.mean(train_res), np.mean(test_res)])

    plt.figure(10, dpi=1200)
    plt.plot([1, 2, 4, 6, 8, 10, 16, 32], np.array(knn_results)[:, 0], '-o', c='red', label='Train')
    plt.plot([1, 2, 4, 6, 8, 10, 16, 32], np.array(knn_results)[:, 1], '-o', c='blue', label='Test')
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Neighbors")
    plt.legend(loc='upper left')
    plt.show()
    print("Done!")


# ===================================================================================================
def svm_classifier(X, Y, Folds):
    print("SVM training started...")
    C_val = [0.0001, 1, 2, 4]
    kernel = ['linear', 'poly', 'rbf']
    svm_results = []
    for k in kernel:
        for c in C_val:
            print("Kernel: {}, C_val: {}".format(k, c))
            train_res = []
            test_res = []
            for train_index, test_index in Folds.split(X, Y):
                svm = SVC(kernel=k, C=c, random_state=SEED)
                svm.fit(X[train_index, :], Y[train_index])
                train_res.append(accuracy_score(Y[train_index], svm.predict(X[train_index, :])))
                test_res.append(accuracy_score(Y[test_index], svm.predict(X[test_index, :])))

            print(np.mean(train_res), np.mean(test_res))
            svm_results.append([np.mean(train_res), np.mean(test_res)])



    plt.figure(10, dpi=1200)
    plt.plot(['0.0001', '1', '2', '4'], np.array(svm_results)[0:5, 0], '-o', c='red', label='Linear Train')
    plt.plot(['0.0001', '1', '2', '4'], np.array(svm_results)[0:5, 1], '-o', c='blue', label='Linear Test')

    plt.plot(['0.0001', '1', '2', '4'], np.array(svm_results)[5:10, 0], '-o', c='m', label='Poly Train')
    plt.plot(['0.0001', '1', '2', '4'], np.array(svm_results)[5:10, 1], '-o', c='teal', label='Poly Test')

    plt.plot(['0.0001', '1', '2', '4'], np.array(svm_results)[10:15, 0], '-o', c='darkgreen', label='RBF Train')
    plt.plot(['0.0001', '1', '2', '4'], np.array(svm_results)[10:15, 1], '-o', c='coral', label='RBF Test')

    plt.ylabel("Accuracy")
    plt.xlabel("Number of Neighbors")
    plt.legend(loc='upper left')
    plt.show()
    print("Done!")


# ===================================================================================================
def decisionTree(X, Y, Folds):
    print("Decision tree training started...")
    max_depth = [3, 5, 8, 12, 16, 20, None]
    decision_tree_results = []
    for depth in max_depth:
        train_res = []
        test_res = []
        for train_index, test_index in Folds.split(X, Y):
            tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
            tree.fit(X[train_index, :], Y[train_index])
            train_res.append(accuracy_score(Y[train_index], tree.predict(X[train_index, :])))
            test_res.append(accuracy_score(Y[test_index], tree.predict(X[test_index, :])))

        print(np.mean(train_res), np.mean(test_res))
        decision_tree_results.append([np.mean(train_res), np.mean(test_res)])

    plt.figure(10, dpi=1200)
    plt.plot(['3', '5', '8', '12', '16', '20', 'None'], np.array(decision_tree_results)[:, 0], '-o', c='red', label='Train')
    plt.plot(['3', '5', '8', '12', '16', '20', 'None'], np.array(decision_tree_results)[:, 1], '-o', c='blue', label='Test')
    plt.ylabel("Accuracy")
    plt.xlabel("Maximum Depth")
    plt.legend(loc='upper left')
    plt.show()
    print("Done!")


# ===================================================================================================
def RandomForest(X, Y, Folds):
    print("RFC training started...")
    num_trees = [10, 20, 40, 80]
    max_depth = [5, 10, 15, 20, None]
    RFC_results = []
    for n_trees in num_trees:
        for depth in max_depth:
            print("Num Trees: {}, Max Depth: {}".format(n_trees, depth))
            train_res = []
            test_res = []
            for train_index, test_index in Folds.split(X, Y):
                rfc = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
                rfc.fit(X[train_index, :], Y[train_index])
                train_res.append(accuracy_score(Y[train_index], rfc.predict(X[train_index, :])))
                test_res.append(accuracy_score(Y[test_index], rfc.predict(X[test_index, :])))

            print(np.mean(train_res), np.mean(test_res))
            RFC_results.append([np.mean(train_res), np.mean(test_res)])

    plt.figure(dpi=1200)
    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[0:5, 0], '-o', c='red', label='N = 10 Train')
    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[0:5, 1], '-o', c='blue', label='N = 10 Test')

    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[5:10, 0], '-o', c='m', label='N = 20 Train')
    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[5:10, 1], '-o', c='teal', label='N = 20 Test')

    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[10:15, 0], '-o', c='darkgreen', label='N = 40 Train')
    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[10:15, 1], '-o', c='coral', label='N = 40 Test')

    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[15:20, 0], '-o', c='pink', label='N = 80 Train')
    plt.plot(['5', '10', '15', '20', 'None'], np.array(RFC_results)[15:20, 1], '-o', c='purple', label='N = 80 Test')

    plt.ylabel("Accuracy")
    plt.xlabel("Maximum Depth")
    plt.legend(loc='upper left')
    plt.show()
    print("Done!")


# ===================================================================================================
def main():
    if HARD_START:
        df_1 = pd.read_csv("mitbih_train.csv", header=None)
        df_2 = pd.read_csv("mitbih_test.csv", header=None)
        df_3 = pd.read_csv("ptbdb_abnormal.csv", header=None)

        main_df = pd.concat([df_1, df_2, df_3])
        preprocessed_df = preprocess(main_df)
    else:
        preprocessed_df = pd.read_csv('preprocessed.csv', header=None)

    preprocessed_df['Type'] = preprocessed_df[16]
    preprocessed_df.drop([16], axis=1, inplace=True)

    X = preprocessed_df.iloc[:, :-1].values
    Y = preprocessed_df['Type'].values

    k_fold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    X = selectTopFeatures(X, Y, k_fold, 10)

    KNN(X, Y, k_fold)
    svm_classifier(X, Y, k_fold)
    decisionTree(X, Y, k_fold)
    RandomForest(X, Y, k_fold)

    # plt.figure(figsize=(8.5, 8.5), dpi=1200)
    # tmp = pd.DataFrame(StandardScaler().fit_transform(preprocessed_df.iloc[:, :-1]))
    # tmp['Type'] = preprocessed_df['Type']
    # data = tmp.sample(frac=0.03, random_state=SEED)
    # print(data)
    # data = pd.melt(data.iloc[:, :], id_vars='Type', var_name="features", value_name='value')
    # sns.swarmplot(x="features", y="value", hue='Type', data=data)
    # plt.show()
    # plt.savefig('s_plot.png')

    # f, ax = plt.subplots(figsize=(18, 18), dpi=1200)
    # sns.heatmap(preprocessed_df.iloc[:, :-1].corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    # plt.show()


if __name__=="__main__":main()
