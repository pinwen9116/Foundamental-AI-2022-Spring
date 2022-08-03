from argparse import ArgumentParser
from typing import Tuple, Union, List, Any
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def data_preprocessing(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return the preprocessed data (X_train, X_test, y_train, y_test). 
    You will need to remove the "Email No." column since it is not a valid feature.
    """
    data_X = data.drop(['Email No.'], axis=1)
    data_X = data_X.drop(['Prediction'], axis=1)
    data_y = data.loc[:, ['Prediction']]
    X_train, X_test = train_test_split(data_X, test_size=0.2, random_state=0, shuffle=False)
    y_train, y_test = train_test_split(data_y, test_size=0.2, random_state =0, shuffle=False)
    return X_train, X_test, y_train, y_test
    raise NotImplementedError

class Tree:
    def __init__(self,data_index, left = None, right = None, feature = None, split = None, out = None):
        self.data_index = data_index
        self.left = left
        self.right = right
        self.feature = feature
        self.split = split
        self.out = out              # 0 or 1

class DecisionTree:
    "Add more of your code here if you want to"
    def __init__(self, tree=[]):
        self.tree = []
    # Reference:
    # https://zhuanlan.zhihu.com/p/32164933?msclkid=637b0979ce7111eca870fd6379d140b3
    # https://github.com/L-ear/RandomForest/blob/master/RandomForest/DecisionTree.py
    #
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        "Fit the model with training data"
        def gini(s):
            p = np.array(s.value_counts(True))
            return 1-np.sum(np.square(p))

        def minigini(data):
            res =[] # gini, feature, split
            S = data.shape[0]
            for feature in np.arange(1, data.shape[1]):
                gini_list = []
                s = data.iloc[:, [0, feature]]
                s = s.sort_values(s.columns[1])
                for i in np.arange(S-1):
                    if  s.iloc[i, 1] == s.iloc[i+1, 1]:
                        continue
                    else:
                        S1 = i + 1
                        S2 = S - S1
                        s1 = data.iloc[:(i + 1), 0]
                        s2 = data.iloc[(i+ 1):, 0]
                        gini_list.append(((S1*gini(s1) + S2*gini(s2))/S,s.iloc[i,1]))
                if gini_list:
                    Gini_min,split = min(gini_list,key=lambda x:x[0])
                    res.append((Gini_min,feature,split))

            if res:
                _, feature, split = min(res, key=lambda x:x[0])
                return (data.columns[feature], split)
            else:
                return None

        def split_leaf(leaf):
            data = X.loc[leaf.data_index]
            res = minigini(data)
            if not res:
                leaf.out = data.iloc[:,0].mode()[0]
                #print("OUT:", leaf.out)
                return None
            feature, split = res
            leaf.feature, leaf.split = feature, split 
            index_l = data[data[feature] <= split].index
            index_r = data[data[feature] > split].index
            left = Tree(index_l)                                     
            right = Tree(index_r)
            return left, right
    
        self.__init__()
        y_list = y['Prediction'].tolist()
        X.insert(loc=0,column='Prediction', value=y_list)
        root = Tree(X.index)
        self.tree.append(root)
        current_tree = 0
        last_tree = 0
        while True:
            #print(current_tree, last_tree)
            res = split_leaf(self.tree[current_tree])
            if res:
                self.tree.extend(res)
                self.tree[current_tree].left = last_tree + 1
                self.tree[current_tree].right = last_tree +2
                current_tree+=1
                last_tree+=2
            elif last_tree == current_tree:
                # no more leaves
                break
            else:
                current_tree+=1
        X = X.drop(['Prediction'], axis=1)
        return

        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> Any:
        "Make predictions for the testing data"
        len_y = len(X.index)
        pre_y = pd.Series([0]*len_y, index=X.index)
        for i in range(len_y):
            y = X.iloc[i]
            j = 0
            while True:
                if self.tree[j].out != None:
                    res = self.tree[j].out
                    break
                if y[self.tree[j].feature] <= self.tree[j].split:
                    j = self.tree[j].left
                else:
                    j = self.tree[j].right
            pre_y.iloc[i] = res

        return pre_y

        raise NotImplementedError


class RandomForest:
    "Add more of your code here if you want to"
    def __init__(self, seed: int = 42, num_trees: int = 5, forest=[]):
        self.num_trees = num_trees
        self.forest = []
        np.random.seed(seed)

    def bagging(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "DO NOT modify this function. This function is deliberately given to make your result reproducible."
        index = np.random.randint(0, X.shape[0], int(X.shape[0] / 2))
        return X.iloc[index, :], y.iloc[index]

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.__init__(num_trees=self.num_trees)
        for i in range(self.num_trees):
            X_use = X.copy(deep=True)
            y_use = y.copy(deep=True)
            X_test, y_test = self.bagging(X_use, y_use)
            X_test = X_test.groupby(X_test.index).first()
            y_test = y_test.groupby(y_test.index).first()
            treei = DecisionTree()
            treei.fit(X_test, y_test)
            self.forest.append(treei)
        return

        raise NotImplementedError

    def predict(self, X) -> Any:
        len_y = len(X.index)
        pre_y = pd.Series([0]*len_y, index=X.index)
        res  = [0] * self.num_trees
        for i in range(len_y):
            y = X.iloc[i]
            for k in range(self.num_trees):
                j = 0
                while True:
                    if self.forest[k].tree[j].out != None:
                        res[k] = self.forest[k].tree[j].out
                        break
                    if y[self.forest[k].tree[j].feature] <= self.forest[k].tree[j].split:
                        j = self.forest[k].tree[j].left
                    else:
                        j = self.forest[k].tree[j].right
            pre_y.iloc[i] = max(res, key=res.count)

        return pre_y
        raise NotImplementedError

from sklearn import metrics
from sklearn.metrics import confusion_matrix
def accuracy_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the accuracy score
    """
    lens = len(y_pred)
    label = y_label['Prediction'].tolist()
    pred = y_pred.tolist()
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(lens):
        if pred[i] == 1 and label[i] == 1:
            tp+=1
        if pred[i] == 1 and label[i] == -1:
            fp+=1
        if pred[i] == -1 and label[i] == -1:
            tn+=1
        if pred[i] == -1 and label[i] == 1:
            fn+=1
    return (tp + tn) / (tp + tn + fp + fn)
    raise NotImplementedError
    
def f1_score(y_pred: Any, y_label: Any) -> float:
    """
    y_pred: (1d array-like) your prediction
    y_label: (1d array-like) the groundtruth label
    Return the F1 score
    """
    lens = len(y_pred)
    label = y_label['Prediction'].tolist()
    pred = y_pred.tolist()
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(lens):
        if pred[i] == 1 and label[i] == 1:
            tp+=1
        elif pred[i] == 1 and label[i] == -1:
            fp+=1
        elif pred[i] == -1 and label[i] == -1:
            tn+=1
        elif pred[i] == -1 and label[i] == 1:
            fn+=1
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp/(tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall)/(precision + recall)
    raise NotImplementedError

def cross_validation(model: Union[LogisticRegression, DecisionTree, RandomForest], X: pd.DataFrame, y: pd.DataFrame, folds: int = 5) -> Tuple[float, float]:
    """
    Test the generalizability of the model with 5-fold cross validation
    Return the mean accuracy and F1 score
    """
    block_size = int(len(X.index) / folds)
    best_acc, best_f1 = 0.0, 0.0
    # split data
    datas = []
    datas.append(X.iloc[:block_size, :])
    datas.append(X.iloc[block_size:block_size*2, :])
    datas.append(X.iloc[2*block_size:block_size*3, :])
    datas.append(X.iloc[3*block_size:block_size*4, :])
    datas.append(X.iloc[4*block_size:block_size*5, :])
    labels = []
    labels.append(y.iloc[:block_size, :])
    labels.append(y.iloc[block_size:block_size*2, :])
    labels.append(y.iloc[2*block_size:block_size*3, :])
    labels.append(y.iloc[3*block_size:block_size*4, :])
    labels.append(y.iloc[4*block_size:block_size*5, :])
    
    for i in range(folds):
        X_1 = pd.DataFrame()
        label_1 = pd.DataFrame()
        for j in range(folds):
            if i != j:
                X_1 = pd.concat([datas[j],X_1])
                label_1 = pd.concat([labels[j], label_1])
        model.fit(X_1, label_1)
        y_pre = model.predict(datas[i])

        # cal score
        acc = accuracy_score(y_pre, labels[i])
        f1 = f1_score(y_pre, labels[i])
        print(acc, f1)
        if f1 > best_f1:
            best_acc = acc
            best_f1 = f1
    best_acc = float(best_acc)
    best_f1 = float(best_f1)
    return tuple([best_acc , best_f1])
    raise NotImplementedError

def tune_random_forest(choices: List[int], X: pd.DataFrame, y: pd.DataFrame, folds: int = 5) -> int:
    """
    choices: List of candidates for the number of decision trees in the random forest
    Return the best choice
    """
    block_size = int(len(X.index) / folds)
    best_acc, best_f1 = 0.0, 0.0
    # split data
    datas = []
    datas.append(X.iloc[:block_size, :])
    datas.append(X.iloc[block_size:block_size*2, :])
    datas.append(X.iloc[2*block_size:block_size*3, :])
    datas.append(X.iloc[3*block_size:block_size*4, :])
    datas.append(X.iloc[4*block_size:block_size*5, :])
    labels = []
    labels.append(y.iloc[:block_size, :])
    labels.append(y.iloc[block_size:block_size*2, :])
    labels.append(y.iloc[2*block_size:block_size*3, :])
    labels.append(y.iloc[3*block_size:block_size*4, :])
    labels.append(y.iloc[4*block_size:block_size*5, :])
    best_f1 = 0.0
    best_num = 0
    length = len(choices)
    for i in range(length):
        for j in range(folds):
            X_1 = pd.DataFrame()
            label_1 = pd.DataFrame()
            for k in range(folds):
                if j != k:
                    X_1 = pd.concat([datas[k],X_1])
                    label_1 = pd.concat([labels[k], label_1])
            forest = RandomForest(num_trees=choices[i])
            forest.fit(X_1, label_1)
            y_pre = forest.predict(datas[j])

            # cal score
            acc = accuracy_score(y_pre, labels[j])
            f1 = f1_score(y_pre, labels[j])
            print(acc, f1)
            if f1 > best_f1:
                best_f1 = f1
                best_num = choices[i]

    return best_num
    raise NotImplementedError

def main(args):
    """
    This function is provided as a head start
    TA will use his own main function at test time.
    """
    data = pd.read_csv(args.data_path)
    print(data.head())
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    
    logistic_regression = LogisticRegression(solver='liblinear', max_iter=500)
    decision_tree = DecisionTree()
    random_forest = RandomForest()
    models = [logistic_regression, decision_tree, random_forest]
    best_f1, best_model = -1, None
    for model in models:
        accuracy, f1 = cross_validation(model, X_train, y_train, 5)
        print(accuracy, f1)
        if f1 > best_f1:
            best_f1, best_model = f1, model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    print(accuracy_score(y_pred, y_test), f1_score(y_pred, y_test))
    
    print(tune_random_forest([5, 11, 17], X_train, y_train))

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./emails.csv')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main(parse_arguments())