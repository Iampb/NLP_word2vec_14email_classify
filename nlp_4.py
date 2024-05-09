import zipfile
import os
import jieba
from gensim.models.word2vec import Word2Vec,LineSentence
import gensim
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import warnings
import time

warnings.filterwarnings("ignore")



#  在本地做的作业，使用直接解压好了文件，这里不使用了
# def unzip_zipfile(zipfile_path, extract_dir):
#     with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
#         zip_ref.extractall(extract_dir)
# unzip_zipfile('D:/THUCNews_new',"./")


def getdata(path):
    a = []
    email_txt = os.listdir(os.path.join("THUCNews/",path))
    # for category in data[]:
    #     email_txt = os.listdir(os.path.join(path, category))
    if email_txt == 'constellation':
        for txt in email_txt:
            with open(os.path.join("THUCNews/", path, txt), 'r', encoding='utf-8') as email:
                txt = email.read()
                a.append(txt)
    else:
        for txt in email_txt[:7588]:
            with open(os.path.join("THUCNews/",path,  txt),'r', encoding='utf-8') as email:
                txt = email.read()
                a.append(txt)
    return a


def normalize_corpus(text):
    text = text.replace("\n","")
    text = text.replace(" ", "")
    text = text.replace("\u3000", "")
    text = text.replace("\xa0", "")
    text = jieba.cut(text, cut_all = False, HMM = True)
    text = list(text)
    with open('stop_word.txt', 'r', encoding='utf-8') as stop_word:
        line = stop_word.readlines()
        stop_wordd = []
        for i in line:
            i = i.replace("\n","")
            stop_wordd.append(i)

        for i in stop_wordd:
            for j in text:
                if i == j:
                    text.remove(j)
    return text


def Word2Vec_extractor(text):
    model = Word2Vec(text,sg = 0,vector_size=50)
    model.save('model/word_vec.model')
    model = Word2Vec.load('model/word_vec.model')
    words = model.wv.index_to_key
    # print(words)
    # print(len(words))
    X = model.wv.vectors
    return X

def Word2Vec_extractor2(text):
    model = Word2Vec(text,sg = 0,vector_size=100)
    model.save('model/word_vec.model2')
    model = Word2Vec.load('model/word_vec.model2')
    words = model.wv.index_to_key
    X = model.wv.vectors
    return X

def Word2Vec_extractor3(text):
    model = Word2Vec(text,sg = 1,vector_size=50)
    model.save('model/word_vec.model3')
    model = Word2Vec.load('model/word_vec.model3')
    words = model.wv.index_to_key
    X = model.wv.vectors
    return X

def Word2Vec_extractor4(text):
    model = Word2Vec(text,sg = 1,vector_size=100)
    model.save('model/word_vec.model4')
    model = Word2Vec.load('model/word_vec.model4')
    words = model.wv.index_to_key
    X = model.wv.vectors
    return X

def NB(X_train,X_test,y_train,y_test):
    clf = GaussianNB()  # 朴素贝叶斯
    clf.fit(X_train, y_train)
    nb_pred_word = clf.predict(X_test)
    print("\t朴素贝叶斯:")
    print("\tPrecision: ", metrics.precision_score(y_test, nb_pred_word, average='macro'))
    print("\tRecall: ", metrics.recall_score(y_test, nb_pred_word, average='macro'))

def LR(X_train,X_test,y_train,y_test):
    lr = LogisticRegression(solver='liblinear')  # 逻辑回归
    lr.fit(X_train, y_train)
    linear_pred_word = lr.predict(X_test)
    print("\t逻辑回归:")
    print("\tPrecision: ", metrics.precision_score(y_test, linear_pred_word, average='macro'))
    print("\tRecall: ", metrics.recall_score(y_test, linear_pred_word, average='macro'))

def SVM(X_train,X_test,y_train,y_test):
    SVM = svm.LinearSVC()  # SVM
    SVM.fit(X_train, y_train)
    svm_pred_word = SVM.predict(X_test)
    print("\t支持向量机:")
    print("\tPrecision: ", metrics.precision_score(y_test, svm_pred_word, average='macro'))
    print("\tRecall: ", metrics.recall_score(y_test, svm_pred_word, average='macro'))

def MLP(X_train,X_test,y_train,y_test):
    mlp = MLPClassifier(solver='lbfgs', max_iter=500)
    mlp.fit(X_train, y_train)
    mlp_pred_word = mlp.predict(X_test)
    print("\tMLP:")
    print("\tPrecision: ", metrics.precision_score(y_test, mlp_pred_word, average='macro'))
    print("\tRecall: ", metrics.recall_score(y_test, mlp_pred_word, average='macro'))

start = time.time()
path = os.listdir("THUCNews")
email_list,email_label,real_email_list = [],[],[]
email_label2,real_email_list2 = [],[]
email_label3,real_email_list3 = [],[]
email_label4,real_email_list4 = [],[]

for k in path:
    email = getdata(k)
    for i in email:
        txt = normalize_corpus(i)
        email_list.append(txt)
    # X_word2vec = Word2Vec_extractor(email_list)
    X_word2vec2 = Word2Vec_extractor2(email_list)
    X_word2vec3 = Word2Vec_extractor3(email_list)
    X_word2vec4 = Word2Vec_extractor4(email_list)
    # for l in X_word2vec:
    #     email_label.append(k)
    #     real_email_list.append(l)
    for l in X_word2vec2:
        email_label2.append(k)
        real_email_list2.append(l)
    for l in X_word2vec3:
        email_label3.append(k)
        real_email_list3.append(l)
    for l in X_word2vec4:
        email_label4.append(k)
        real_email_list4.append(l)


# X_train_word1, X_test_word1, y_train_word1, y_test_word1 = train_test_split(real_email_list ,email_label, test_size=0.2, random_state=0)
X_train_word2, X_test_word2, y_train_word2, y_test_word2 = train_test_split(real_email_list2 ,email_label2, test_size=0.2, random_state=0)
X_train_word3, X_test_word3, y_train_word3, y_test_word3 = train_test_split(real_email_list3 ,email_label3, test_size=0.2, random_state=0)
X_train_word4, X_test_word4, y_train_word4, y_test_word4 = train_test_split(real_email_list4 ,email_label4, test_size=0.2, random_state=0)

# print(X_train_word)
# print(y_train_word)
# scaler = MinMaxScaler()
# scaler.fit(X_train_word)
# X_train_scaled = scaler.transform(X_train_word)
# X_test_scaled = scaler.transform(X_test_word)

#-----------------------------------------------开始训练-------------------------------------------
#----------使用cbow算法------------------------
# print("使用cbow,vector_size为50")
# NB(X_train_word1,X_test_word1,y_train_word1,y_test_word1)
# LR(X_train_word1,X_test_word1,y_train_word1,y_test_word1)
# SVM(X_train_word1,X_test_word1,y_train_word1,y_test_word1)
# MLP(X_train_word1,X_test_word1,y_train_word1,y_test_word1)


#----------------使用cbow,vector_size为100----------------------------------
# print()
# print("使用cbow,vector_size为100")
# NB(X_train_word2,X_test_word2,y_train_word2,y_test_word2)
# LR(X_train_word2,X_test_word2,y_train_word2,y_test_word2)
# SVM(X_train_word2,X_test_word2,y_train_word2,y_test_word2)
MLP(X_train_word2,X_test_word2,y_train_word2,y_test_word2)

#----------------使用skip-gram,vector_size为50----------------------------------
print()
print("使用skip-gram,vector_size为50")
NB(X_train_word3,X_test_word3,y_train_word3,y_test_word3)
LR(X_train_word3,X_test_word3,y_train_word3,y_test_word3)
SVM(X_train_word3,X_test_word3,y_train_word3,y_test_word3)
MLP(X_train_word3,X_test_word3,y_train_word3,y_test_word3)


#----------------使用skip-gram,vector_size为100----------------------------------
print()
print("使用skip-gram,vector_size为100")
NB(X_train_word4,X_test_word4,y_train_word4,y_test_word4)
LR(X_train_word4,X_test_word4,y_train_word4,y_test_word4)
SVM(X_train_word4,X_test_word4,y_train_word4,y_test_word4)
MLP(X_train_word4,X_test_word4,y_train_word4,y_test_word4)


end = time.time()
times = end - start
print("运行时间：%.2f 秒" %times)