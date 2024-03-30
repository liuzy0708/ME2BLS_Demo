import argparse
from utils import get_stream, get_classifier, get_strategy
from pprint import pprint
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
t1 = time.time()

#IDE版本
class set_up():
    def __init__(self, dataset_name, n_round, train_size, strategy_name,
                 classifier_name, label_ratio, chunk_size):
        self.dataset_name = dataset_name
        self.n_round = n_round
        self.train_size = train_size
        self.strategy_name = strategy_name
        self.classifier_name = classifier_name
        self.label_ratio = label_ratio
        self.chunk_size = chunk_size

    def show_setup(self):
        print("dataset_name:", self.dataset_name)
        print("n_round:", self.n_round)
        print("train_size:",self.train_size)
        print("strategy_name:", self.strategy_name)
        print("classifier_name:", self.classifier_name)
        print("chunk_size:", self.chunk_size) #instance-level中chunk_size==1

args = set_up(dataset_name="Hyperplane.csv", n_round=2, train_size=200, strategy_name="RandomSampling", classifier_name="NaiveBayes", label_ratio=0.2, chunk_size=1)
pprint(vars(args))
print()

#%%

accuracy_round = []
f1_round = []

for i in range(args.n_round):
    print('round{}'.format(i+1))
    #%%
    #数据集和策略准备
    stream = get_stream(args.dataset_name)                   # load dataset
    classifier = get_classifier(args.classifier_name)
    x_train, y_train = stream.next_sample(args.train_size)
    classifier.fit(x_train, y_train)
    label_strategy_class = get_strategy(args.strategy_name)  # load strategy，考虑到策略里面会有一些存储器，所以需要使用类而不是函数
    label_strategy = label_strategy_class(args.label_ratio)
    #%%

    #%%储存预测结果和真实值
    y_predict_list = np.array([])
    y_true_list = np.array([])
    accuracy_vary_list = np.array([])
    f1_vary_list = np.array([])
    #%%

    count = 0
    while stream.has_more_samples():
        count = count + args.chunk_size
        print(count)
        x, y = stream.next_sample(args.chunk_size)
        y_predict = classifier.predict(x)
        y_predict_list = np.append(y_predict_list, y_predict)
        y_true_list = np.append(y_true_list, y)
        # accuracy_vary_list.append(accuracy_score(y_true_list, y_predict_list))
        # f1_vary_list.append(f1_score(y_true_list, y_predict_list, average='macro'))

        #将label做成列表的形式是为了使instance-level和chunk-level形式一致
        islabel = label_strategy.get_label(x, classifier)
        for i in range(len(islabel)):
            if islabel[i] == True:
                classifier.partial_fit(np.array([x[i,:]]), np.array([y[i]]))

    print('round{}: acc={}, f1={}'.format(i+1, accuracy_score(y_true_list, y_predict_list), f1_score(y_true_list, y_predict_list, average='macro')))
    accuracy_round.append(accuracy_score(y_true_list, y_predict_list))
    f1_round.append(f1_score(y_true_list, y_predict_list, average='macro'))


print("overall accuracy = {} ± {}".format(np.mean(accuracy_round), np.std(accuracy_round)))
print("overall f1 = {} ± {}".format(np.mean(f1_round), np.std(f1_round)))
t2 = time.time()
print('the total time is {} s'.format(t2 - t1))

