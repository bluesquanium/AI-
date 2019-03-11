import tensorflow as tf
#서버에서 실행할 때
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # put assigned gpu number
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.19
session = tf.Session(config=config)
import numpy as np
import pandas as pd
import json

#MinMaxScaler 함수를 구현하여 사용하였다.
def MinMaxScaler(data) :
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-5)


def jsonRead(li_input, frequency, str_target):
    # 함수 내의 주석 예는 영화 제작 회사를 추출할 때로 예를 들음
    i = 0
    li_items = []
    u_li_items = []  # 얘는 unique한 회사이름 저장하기 위해
    # while문 통해 돌면서 회사 추출해서 저장
    while True:
        if i == len(li_input):  # 반복문이 끝나는 조건 : 끝까지 다 봤을 때
            break

        try:
            j = (li_input[i][0])
            j = j.replace("'", '"')  # json.loads 할 때 ''로 되있으면 안되고 ""로 바꿔야함
            j = json.loads(j)
        except Exception as e:
            # print(e)
            j = li_input[i][0]

            index = 0
            while index != len(j):
                if j[index] == '"':
                    d = 1
                    while j[index + d] != '"':
                        if j[index + d] == "'":  # '가 나오면 공백으로 대체
                            j = j[:index + d] + ' ' + j[index + d + 1:]
                        d += 1
                    index += d
                if j[index] == "'":
                    d = 1
                    while j[index + d] != "'":
                        if j[index + d] == '"':  # '가 나오면 공백으로 대체
                            j = j[:index + d] + ' ' + j[index + d + 1:]
                        d += 1
                    index += d

                index += 1
            j = j.replace("'", '"')
            try:
                j = json.loads(j)
            except Exception as e:
                print(e)
                print(i)
                print(li_input[i][0])
                print(j)

            # for k in range(len(j)) :
            #    print(j[k]['name'])
            # break

        temp = []
        for k in range(len(j)):
            temp.append(j[k][str_target])
            u_li_items.append(j[k][str_target])
        # movie_companies.append(', '.join(temp)) #판다스저장용
        li_items.append(temp)

        i += 1

    u_li_items = keywords_frequency(u_li_items, frequency)  # frequency 횟수 이상 나온 것들만 저장
    # u_movie_companies = set(u_movie_companies)
    # df_movie_companies = pd.DataFrame(movie_companies) #판다스 저장용

    return li_items, u_li_items


# 이 함수는 jsonRead를 통해 정보를 뽑아낸 후 그 정보를 갖고서 one_hot 인코딩을 해주기 위한 함수
def convertOnehot(li_items, u_li_items):
    li_item_dict = {}
    num = 0
    for item in u_li_items:
        li_item_dict[item] = num
        num += 1

    for i in range(0, len(li_items)):
        for j in range(0, len(li_items[i])):
            if li_items[i][j] in li_item_dict:
                li_items[i][j] = li_item_dict[li_items[i][j]]
            else:
                li_items[i][j] = -1

    li_item_one_hot = np.zeros((len(li_items), len(u_li_items)), dtype=int)
    for i in range(0, len(li_item_one_hot)):
        for j in range(0, len(li_items[i])):
            if li_items[i][j] != -1:
                li_item_one_hot[i][li_items[i][j]] = 1

    # print(li_item_one_hot.shape)
    # print(li_item_one_hot)
    return li_items, li_item_one_hot

#키워드가 등장하는 횟수를 세서 등장횟수 적은 키워드들을 제거한다.
def keywords_frequency(li, frequency) :
    # 각 키워드가 나온 횟수를 딕셔너리로 만듦.
    # input : 키워드가 들어있는 리스트(li), 등장 횟수 몇번 이상으로 자를지(frequency)
    freq = {}
    for i in li :
        if i in freq.keys() :
            freq[i] += 1
        else :
            freq[i] = 1

    for k, v in freq.items() :
        if v > frequency :
            pass
        else :
            freq[k] = -1

    freq2 = {}
    for k, v in freq.items() :
        if v!= -1 :
            freq2[k] = v

    return freq2

movie_train_df = pd.read_csv('../../../assets/movie_train.csv', header = None)
movie_test_df = pd.read_csv('../../../assets/movie_test.csv', header = None)

#예산, 수익률 전처리
budgets = np.array(movie_train_df[[2]])
test_budgets = np.array(movie_test_df[[2]])
regul_budgets = MinMaxScaler(budgets) # 정규화를 해줌
test_regul_budgets = MinMaxScaler(budgets)
profits = np.array(movie_train_df[[15]])
test_profits = np.array(movie_test_df[[15]])

profit_rate = []
for i in range(len(budgets)) :
    temp = []
    temp.append(round(float(regul_budgets[i]),3))
    if budgets[i] == 0.0 :
        temp.append(0)
    else :
        temp.append(round((100*float(profits[i]))/(float(budgets[i])),3))
    profit_rate.append(temp)
profit_rate = np.array(profit_rate)
test_profit_rate = []
for i in range(len(test_budgets)) :
    temp = []
    temp.append(round(float(test_regul_budgets[i]),3))
    if test_budgets[i] == 0.0 :
        temp.append(0)
    else :
        temp.append(round((100*float(test_profits[i]))/(float(test_budgets[i])),3))
    test_profit_rate.append(temp)
test_profit_rate = np.array(test_profit_rate)

#인기도 (popularity)
popularity = np.array(movie_train_df[10])
popularity = np.reshape(popularity, (len(popularity),1))
popularity = MinMaxScaler(popularity) # 정규화 해줌
test_popularity = np.array(movie_test_df[10])
test_popularity = np.reshape(test_popularity, (len(test_popularity),1))
test_popularity = MinMaxScaler(test_popularity)

#사용 언어를 one_hot으로
languages = np.array(movie_train_df[7])
test_languages = np.array(movie_test_df[7])
# u_languages, lang_dict는 test, train에 대해 하나로 사용
u_languages = set(languages)
lang_dict = {}
num = 0
for lang in u_languages : # lang_dict 만들기
    lang_dict[lang] = num
    num += 1
for i in range(0, len(languages)) : # lang_dict 이용해서 스트링을 int로 바꿔주기
    languages[i] = lang_dict[languages[i]]
languages_ont_hot = np.zeros((len(languages), len(u_languages)), dtype=int)
for i in range(0, len(languages_ont_hot)) : # one_hot으로 만들어주기
    languages_ont_hot[i][languages[i]] = 1

for i in range(0, len(test_languages)) : # lang_dict 이용해서 스트링을 int로 바꿔주기
    test_languages[i] = lang_dict[test_languages[i]]
test_languages_ont_hot = np.zeros((len(test_languages), len(u_languages)), dtype=int)
for i in range(0, len(test_languages_ont_hot)) : # one_hot으로 만들어주기
    test_languages_ont_hot[i][test_languages[i]] = 1

#제작 회사 얻어내기
companies = (movie_train_df[[12]]) # company 정보 긁어온다
companies = companies.values.tolist() # 리스트로 변환시켜줌
movie_companies, u_movie_companies = jsonRead(companies, 5, 'name')
movie_companies, company_one_hot = convertOnehot(movie_companies, u_movie_companies)
test_companies = (movie_test_df[[12]]) # company 정보 긁어온다
test_companies = test_companies.values.tolist() # 리스트로 변환시켜줌
test_movie_companies, _ = jsonRead(test_companies, 5, 'name') #u_movie_companies는 train, test 동일한 것을 사용하기에 _로 처리함
test_movie_companies, test_company_one_hot = convertOnehot(test_movie_companies, u_movie_companies)

#장르 얻어내기
genres = movie_train_df[[3]] # genre
genres = genres.values.tolist()
movie_genres, u_movie_genres = jsonRead(genres, 0, 'name')
movie_genres, genre_one_hot = convertOnehot(movie_genres, u_movie_genres)
test_genres = movie_test_df[[3]] # genre
test_genres = test_genres.values.tolist()
test_movie_genres, _ = jsonRead(test_genres, 0, 'name')
test_movie_genres, test_genre_one_hot = convertOnehot(test_movie_genres, u_movie_genres)

x_train = np.concatenate((profit_rate, popularity, genre_one_hot, languages_ont_hot, company_one_hot), axis = 1) # budget, profit, popularity, genre, language, company]
y_train = np.array(movie_train_df[[22]])
x_test = np.concatenate((test_profit_rate, test_popularity, test_genre_one_hot, test_languages_ont_hot, test_company_one_hot), axis = 1) # budget, profit, popularity, genre, language, company]
#y_train = np.array(movie_test_df[[22]])

print(x_train.shape)
print(y_train.shape)

epsilon = 1e-3
#batch normalization 함수
def batch_norm_wrapper(inputs, is_training, decay=0.999) :
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean,
                                pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

# 배치 function
def next_batch(num, data, labels):
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[i] for i in idx]
  labels_shuffle = [labels[i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

#Placeholder들
X = tf.placeholder(tf.float32, [None, 553])
Y = tf.placeholder(tf.float32, [None, 1])
dropout_prob = tf.placeholder(tf.float32)

#핵심 Neural Network
Flat_X = tf.reshape(X, [-1, 553])
W1 = tf.Variable(tf.random_normal([553, 1024], stddev=0.01))
B1 = tf.Variable(tf.constant(0.1, shape=[1024]))
Z1 = tf.matmul(Flat_X, W1)
L1 = tf.nn.sigmoid(Z1+B1)

W1_2 = tf.Variable(tf.random_normal([1024, 1024], stddev=0.01))
B1_2 = tf.Variable(tf.constant(0.1, shape=[1024]))
Z1_2 = tf.matmul(L1, W1_2)
L1_2 = tf.nn.sigmoid(Z1_2+B1_2)

W2 = tf.Variable(tf.random_normal([1024,512], stddev=0.01))
B2 = tf.Variable(tf.constant(0.1, shape=[512]))
Z2 = tf.matmul(L1_2, W2)
L2 = tf.nn.sigmoid(Z2+B2)

W3 = tf.Variable(tf.random_normal([512,256], stddev=0.01))
B3 = tf.Variable(tf.constant(0.1, shape=[256]))
Z3 = tf.matmul(L2, W3)
L3 = tf.nn.sigmoid(Z3+B3)

W4 = tf.Variable(tf.random_normal([256,1], stddev=0.01))
B4 = tf.Variable(tf.constant(0.1, shape=[1]))
Z4 = tf.matmul(L3, W4)
#model = tf.nn.sigmoid(Z4+B4)
model = Z4 + B4

cost = tf.reduce_mean(tf.reduce_sum(tf.square(model-Y),[1]))
train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

#is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#초기화 및 세션 실행
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# batch 설정
batch_size = 200
total_batch = int(len(x_train) / batch_size)

#트레이닝 Part
print("Training Start!")
for epoch in range(2000):

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)

        sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys, dropout_prob: 0.75})

    if (epoch+1)%50 == 0 :
        print('Epoch:', '%04d' % (epoch+1), 'Cost:', sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, dropout_prob: 1.0}))
print('Training Done!')

#테스트 시작
#batch_size 위와 동일

'''
t_cost = 0.0
for i in range(total_batch):
    batch_xs, batch_ys = next_batch(batch_size, x_test, y_test)
    cost_val = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})
    t_cost += cost_val
    print(cost_val)
t_cost = t_cost / total_batch
print('RMSE : ', round(t_cost,3))
'''

for i in range(len(x_test)) :
    predict_rate = model.eval(session=sess, feed_dict={X: x_test[i:i+1], dropout_prob: 1.0})
    print(i, " , ", "%0.1f" % predict_rate)
