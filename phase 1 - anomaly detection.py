## Libraries-------------------------------------------------------------------
import numpy as np  # کتابخانه‌ای برای انجام محاسبات عددی و عملیات‌های آرایه‌ای
import pandas as pd  # کتابخانه‌ای برای کار با داده‌های جدولی و پردازش داده‌ها
import re  # کتابخانه‌ای برای کار با عبارات منظم و انجام عملیات‌های جستجو و جایگزینی در رشته‌ها
import parsivar  # کتابخانه‌ای برای پردازش زبان فارسی
from parsivar import FindStems  # وارد کردن کلاس FindStems از کتابخانه Parsivar برای ریشه‌یابی کلمات فارسی
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier  # وارد کردن مدل‌های رگرسیون لجستیک، رگرسیون خطی و طبقه‌بند گرادیان نزولی از scikit-learn
from sklearn.metrics import confusion_matrix  # وارد کردن تابع confusion_matrix از scikit-learn برای محاسبه ماتریس درهم‌ریختگی
from sklearn.preprocessing import LabelEncoder  # وارد کردن کلاس LabelEncoder از scikit-learn برای کدگذاری برچسب‌ها
from sklearn.model_selection import train_test_split  # وارد کردن تابع train_test_split از scikit-learn برای تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی


## Read Dataframe-------------------------------------------------------------
data = pd.read_csv("C:/Users/Amir/Desktop/nk.csv")  # خواندن فایل CSV از مسیر مشخص شده و ذخیره آن در DataFrame به نام data
df = pd.DataFrame(data["comment"])  # ایجاد یک DataFrame جدید با ستون "comment" از DataFrame اصلی
df["label"] = data["recommend"]  # اضافه کردن ستون "recommend" از DataFrame اصلی به DataFrame جدید با نام "label"
  
## Dara Cleaning---------------------------------------------------------------
df["label"].value_counts()      #get a summary of data
df1 = df[df['label'] == "recommended"]      #choose recommended comments in df1
df2 = df[df['label'] == "not_recommended"]  #choose not recommended comments in df2
df = pd.concat([df1, df2])    # merge 2 dataframe
sp = df.sample(10000) # get a 10000 random data as sample
C =  list(range(0, 10000))
sp.index = list(range(0,10000))
## Preprocessing---------------------------------------------------------------
stop = pd.read_csv("C:/Users/Amir/Desktop/stop words.csv")  # خواندن فایل CSV حاوی کلمات توقف و ذخیره آن در DataFrame به نام stop
stop = stop["stop"].tolist()  # تبدیل ستون "stop" از DataFrame به لیست

sp['comment'] = sp['comment'].astype(str)  # تبدیل سری "comment" به رشته برای استفاده از تابع split
sp['stopwords'] = sp['comment'].apply(lambda x: len([w for w in x.split() if w in stop]))  # شمارش کلمات توقف در هر کامنت

sp['comment'] = sp['comment'].apply(lambda x: re.sub('[^\D\s]', '', x))  # حذف علائم نگارشی و اعداد از کامنت‌ها

sp['comment'] = sp['comment'].apply(lambda x: (" ").join([FindStems().convert_to_stem(w) for w in x.split()]))  # ریشه‌یابی کلمات در هر کامنت

spamtext = sp[sp['label'] == 'not_recommended']['comment']  # جدا کردن کامنت‌های توصیه نشده
hamtext = sp[sp['label'] == 'recommended']['comment']  # جدا کردن کامنت‌های توصیه شده

spam_no_stop = spamtext.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))  # حذف کلمات توقف از کامنت‌های توصیه نشده
hamw_no_stop = hamtext.apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))  # حذف کلمات توقف از کامنت‌های توصیه شده

spam_word_counts = spam_no_stop.str.split(expand=True).stack().value_counts()  # شمارش تعداد وقوع هر کلمه در کامنت‌های توصیه نشده
ham_word_counts = hamw_no_stop.str.split(expand=True).stack().value_counts()  # شمارش تعداد وقوع هر کلمه در کامنت‌های توصیه شده

spamwords_usable = spam_word_counts[spam_word_counts >= 10]  # انتخاب کلمات توصیه نشده که حداقل 10 بار تکرار شده‌اند
hamwords_usable = ham_word_counts[ham_word_counts >= 10]  # انتخاب کلمات توصیه شده که حداقل 10 بار تکرار شده‌اند
##Data preparing---------------------------------------------------------------


s1=set(spamwords_usable.index)
Union=s1.union(set(hamwords_usable.index))
Union=pd.Series(list(Union))

allfeatures=np.zeros((sp.shape[0],Union.shape[0]))
for i in range(Union.shape[0]):
    word = Union[i]
    allfeatures[:, i] = sp['comment'].apply(lambda x: len(re.findall(r'\b{}\b'.format(re.escape(word)), x)))


Complete_data=pd.concat([sp,pd.DataFrame(allfeatures)],axis=1)
X=Complete_data.iloc[:,2:]
y=Complete_data['label']
from sklearn.preprocessing import scale
X = scale(X)
enc=LabelEncoder()
enc.fit(y)
y = enc.transform(y)
repeat=10

acc_lasso_ham=np.empty(repeat)
acc_lasso_spam=np.empty(repeat)
acc_ridge_ham=np.empty(repeat)
acc_ridge_spam=np.empty(repeat)
acc_elnet_ham=np.empty(repeat)
acc_elnet_spam=np.empty(repeat)


##Fit model--------------------------------------------------------------------

for i in range(repeat):  # تکرار فرآیند به تعداد دفعات مشخص شده در متغیر repeat
    print(i)

    # تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی با نسبت 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # ایجاد مدل‌های رگرسیون لجستیک Lasso و Ridge
    lassologreg = LogisticRegression(C=15, penalty="l1", solver="liblinear")
    ridgelogreg = LogisticRegression(C=15, penalty="l2", solver="liblinear")
    
    # آموزش مدل‌ها با داده‌های آموزشی
    lassologreg.fit(X_train, y_train)
    ridgelogreg.fit(X_train, y_train)
   
    # محاسبه ماتریس درهم‌ریختگی برای مدل‌های Lasso و Ridge
    lasso = confusion_matrix(y_test, lassologreg.predict(X_test))
    ridge = confusion_matrix(y_test, ridgelogreg.predict(X_test))
    
    # محاسبه دقت مدل‌ها برای کلاس‌های Ham و Spam و ذخیره نتایج در آرایه‌های مربوطه
    acc_lasso_ham[i] = lasso[0,0] / sum(lasso[0,:])
    acc_lasso_spam[i] = lasso[1,1] / sum(lasso[1,:])
    acc_ridge_ham[i] = ridge[0,0] / sum(ridge[0,:])
    acc_ridge_spam[i] = ridge[1,1] / sum(ridge[1,:])
   
# چاپ میانگین دقت مدل‌ها برای کلاس‌های Ham و Spam
print('GLM Lasso Ham', '\n', np.mean(acc_lasso_ham))
print('GLM Lasso Spam', '\n', np.mean(acc_lasso_spam))
print('GLM Ridge Ham', '\n', np.mean(acc_ridge_ham))
print('GLM Ridge Spam', '\n', np.mean(acc_ridge_spam))









