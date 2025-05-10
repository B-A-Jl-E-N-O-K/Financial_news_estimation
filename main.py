from threading import Thread, Lock, Event
import time
import random
from bs4 import BeautifulSoup
import requests
import pandas as pd
from pymystem3 import Mystem
import nltk
from nltk.corpus import stopwords
import ir
import psycopg2
from psycopg2 import Error
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import requests
import xml.etree.ElementTree as ET


last_update_datetime = pd.to_datetime('2000-01-01 00:00:00')

mystem = Mystem()
nltk.download('stopwords')
swords = set(stopwords.words('russian'))

companies = {
    'ABRD': 'Абрау-Дюрсо ПАО ао', 
    'AFLT': 'Аэрофлот-росс.авиалин(ПАО)ао', 
    'ALRS': 'АЛРОСА ПАО ао',
    'APTK': 'ПАО "Аптечная сеть 36,6" ао',
    'CHMF': 'Северсталь (ПАО)ао',
    'GAZP': '"Газпром" (ПАО) ао', 
    'GMKN': 'ГМК "Нор.Никель" ПАО ао', 
    'LKOH': 'НК ЛУКОЙЛ (ПАО) - ао',
    'LSRG': 'Группа ЛСР ПАО ао',
    'MGNT': '"Магнит" ПАО ао',
    'MGTS': '"ПАО "МГТС" ао',
    'MOEX': 'ПАО Московская Биржа', 
    'MVID': '"М.видео" ПАО ао',
    'NAUK': 'НПО Наука ао',
    'PIKK': 'ПИК СЗ (ПАО) ао', 
    'PLZL': 'Полюс ПАО ао',
    'RBCM': 'ГК РБК ПАО ао',
    'ROSN': 'ПАО НК Роснефть',
    'RTKM': 'Ростелеком (ПАО) ао.',  
    'SBER': 'Сбербанк России ПАО ао',
    'TATN': 'ПАО "Татнефть" ао',
    'VTBR': 'ао ПАО Банк ВТБ'
    }

for_regex = {
    'ABRD': ['абрау-дюрсо', 'абрау', 'abrd'],
    'ALRS': ['алроса', 'alrs'], 
    'AFLT': ['аэрофлот', 'aflt'],
    'APTK': ['36.6', '36,6', 'aptk'],
    'CHMF': ['cеверсталь', 'chmf'],
    'VTBR': ['втб', 'vtbr'], 
    'GAZP': ['газпром', 'gazp'], 
    'GMKN': ['норникель', 'нор никель', 'норильский никель', 'gmkn'],
    'LKOH': ['лукойл', 'lkoh'],
    'LSRG': ['лср', 'lsr', 'lsrg'], 
    'MGNT': ['магнит', 'magnit', 'mgnt'],
    'MGTS': ['мгтс', 'московская городская телефонная сеть', 'mgts'],
    'MOEX': ['московский биржа', 'мосбиржа', 'мос биржа', 'мос. биржа', 'moex'],
    'MVID': ['мвидео', 'м. видео', 'м.видео', 'mvideo', 'm. video', 'm.video', 'mvid'],
    'NAUK': ['нпо наука', 'nauk'],
    'PIKK': ['pik', 'пик', 'первая ипотечная компания', 'pikk'], 
    'PLZL': ['plzl', 'полюс золото'],
    'RBCM': ['rbc', 'рбк', 'росбизнесконсалтинг', 'ragr'],
    'ROSN': ['роснефть', 'рос нефть', 'rosneft', 'rosn'],
    'RTKM': ['ростелеком', 'рос телеком', 'rtkm'],  
    'SBER': ['/^сбер$/', 'сбербанк', 'sber'],   
    'TATN': ['tatn', 'tatneft', 'татнефть'], 
    'VTBR': ['втб', 'vtb', 'vtbr']
}

for_regex_industry = {
    'finance': ['ипотека', 'sber', '/^сбер$/', 'сбербанк', 'банк', 'кредит', 'депозит', 'мкб', 'московский кредитный банк', 'cbom', 'vtbr', 'втб', 'финансовый', 'финансы', 'биржа', 'moex', 'tcsg', 'тинькофф', 'ценные бумаги', 'ценная бумага', 'облигация', 'облигации', 'паевый', 'фонд', 'котировка'],
    'gold': ['poly', 'plzl', 'полиметалл', 'полюс золото', 'золото', 'pogr', 'petropavlovsk plc', 'petropavlovsk'],
    'ferrous_metallurgy': ['железо', 'сталь', 'северсталь', 'чугун', 'ммк', 'магнитогорский металлургический комбинат', 'magn', 'северсталь', 'chmf', 'nlmk', 'новолипецкий металлургический комбинат', 'металл'],
    'oil_gas':['gazp', 'газпром', 'rosn', 'роснефть', 'сургутнефтегаз', 'sngs', 'газ', 'нефть', 'нефтепровод', 'газопровод', 'татнефть', 'tatn', 'лукойл', 'lkoh', 'транснефть', 'trnfp', 'новатэк', 'nvtk'],
    'non_ferrous_metallurgy': ['rual', 'rusal', 'uc rusal','русал', 'gmkn', 'норникель', 'норильский никель', 'ниель', 'алюминий', 'палладий', 'олово', 'медь', 'глинозем'],
    'electrical networks': ['интер рао', 'irao', 'русгидро', 'hydr', 'элетричество', 'гэс', 'тэц', 'энергетика', 'фск еэс', 'fees'],
    'telecom': ['мтс', 'mtss', 'ростелеком', 'rtkm', 'сеть', 'связь', 'интернет', 'провайдер'],
    'it': ['машинный обучение', 'цифровой технология', 'облако', 'вк', 'мэйл', 'вконтакте', 'mail', 'vk', 'vkco', 'яндекс', 'yandex', 'yndx'],
    'real_estate': ['недвижимость', 'новостройка', 'ипотека', 'лср', 'lsrg', 'пик', 'pikk'],
    'consume_retail': ['напиток', 'вода', 'абрау', 'дюрсо', 'abrd', 'алкоголь', 'вино', 'еда', 'пищевой', 'пища', 'магнит', 'перекресток', 'пятерочка', 'продукт', 'магазин'],
    'tech_market': ['м.видео', 'mvideo', 'm. video', 'dns', 'citilink', 'смартфон', 'ноутбук', 'телевизор', 'бытовой', 'техника'],
    'research': ['исследование', 'наука', 'научный', 'nauk', 'нпо'],
}

industries = for_regex_industry.keys()

try:
    connection = psycopg2.connect(user="postgres",
                                    password="123123",
                                    host="127.0.0.1",
                                    port="5432")
        
    cursor = connection.cursor()
    sql_create_database = 'create database news_estimation_db'

except (Exception, Error) as error:
    print('Error in connection to db:', error)
    raise
finally:
    if connection:
            cursor.close()
            connection.close()


table_name = 'news_est_table'

try:
    connection = psycopg2.connect(user="postgres",
                                    password="123123",
                                    host="127.0.0.1",
                                    port="5432",
                                    database="news_estimation_db")
        
    
    cursor = connection.cursor()
    
    create_table_query = f'''CREATE TABLE IF NOT EXISTS {table_name}
                          (id SERIAL PRIMARY KEY,
                          link varchar(100) NOT NULL,
                          datetime varchar(50) NOT NULL,
                          title_origin TEXT NOT NULL,
                          company varchar(100)); '''
    cursor.execute(create_table_query)

except (Exception, Error) as error:
    print('Error in create table:', error)
    raise
finally:
    if connection:
            cursor.close()


supply_counter = 0
success_supply_counter = 0

def main():
    global supply_counter, success_supply_counter
    while 1:
        n_sec = 20 # 180 seconds interval of taking news
        start_time = time.time()
        is_taken_news = est_news()

        if is_taken_news == 0:

            success_supply_counter += 1

            collecting_time = round(time.time() - start_time)
            print(f"Collecting news time (iteration {supply_counter}): {collecting_time} seconds")
            if collecting_time > 120:
                print(f"Warning: Collecting news time above limit")

        supply_counter += 1
        time.sleep(n_sec)
        if supply_counter % (1) == 0: # 1 week interval

            start_time = time.time()
            update_ml()

            collecting_time = round(time.time() - start_time)
            print(f"Making ML model (iteration {supply_counter}): {collecting_time} seconds")
            if collecting_time > 300:
                print(f"Warning: Making ML model time above limit")
            time.sleep(60)


def est_news():
    global data, current_val, last_update_datetime

    
    urls = load_links() # List of dicts
    urls = to_standard_date(urls) # DataFrame
    urls = urls[urls['datetime'] > last_update_datetime]
    urls.reset_index(drop=True, inplace=True)

    total_news = len(urls)
    if total_news == 0:
        return -1
    start_parsing(parse, parse_from_link_komersant, urls['link'].tolist(), total_news, n_threads=2)
    
    news_df = pd.DataFrame(data)
    news_df = news_df.merge(urls, how='inner', on='link')
    news_df.reset_index(drop=True, inplace=True)


    data.clear()
    data['link'] = []
    data['data_or_ex'] = []
    current_val = 0

    news_df = lemm_file(news_df, mystem, swords) # Text preprocess + lemmatize
    if(news_df is None):
        return -1
    
    # last_update_datetime = news_df['datetime'].max()

    news_df = add_company(news_df)
    news_df = add_industry(news_df) # Last use of text frame 
    ml_ready_df = make_dataset(news_df) # New df for ml
    predict = make_estimation(ml_ready_df)
    result_df = pd.concat([news_df, pd.DataFrame(predict, columns=['class'])], axis=1)

    save_db_df = result_df.loc[:, ['link', 'datetime','title origin', 'company']]
    out_df = result_df.loc[:, ['link', 'datetime', 'title out','company', 'industry', 'class']]
    save_db(save_db_df)
    results_out(out_df)

    return 0
    

### Parsing news ###

def load_links():
    rubrics = {'3': 'economics', '4': 'business', '40':'finance'}
    urls_for_rubrics = [] #List of dicts

    for rubric in rubrics:
        link = f'https://www.kommersant.ru/archive/rubric/{rubric}'
        urls_for_rubrics.append(save_urls(link))
        time.sleep(1 + random.random())
        
    return urls_for_rubrics


def save_urls(link):
    urls = {}
    txt = requests.get(link).text
    soup = BeautifulSoup(txt, 'lxml')
    lst = soup.find_all('article', class_='uho rubric_lenta__item js-article')
    
    for elem in lst:
        urls[str(elem.get('data-article-url'))] = elem.find('p', class_='uho__tag rubric_lenta__item_tag hide_desktop').text.strip()
        
    return urls


def to_standard_date(urls_dicts_list):
    
    df = pd.DataFrame()
    for url in urls_dicts_list:
        row = []
        for link, pub_time in url.items():
            row.append([link, pub_time])
        df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
    
    df.rename(columns={0: 'link', 1: 'datetime'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'],  format='%d.%m.%Y, %H:%M')

    return df
            

def parse_from_link_komersant(link, user_agent):

    
    try:
        with requests.session() as s:
            txt = s.get(link, headers={'User-Agent':user_agent}).text
            soup = BeautifulSoup(txt, 'lxml')
            
            try:
                title = soup.find('h1', class_='doc_header__name js-search-mark').text
            except:
                title = 'No title'
            try:
                announce = soup.find('h2', class_='doc_header__subheader').text
            except:
                announce = 'No announce'
            try:
                try:
                    text = soup.find('p', class_='doc__text doc__intro').text
                    txt = soup.find_all('p', class_='doc__text')
                    text = text + ' ' + ' '.join([el.text for el in txt])
                except:
                    txt = soup.find_all('p', class_='doc__text')
                    text = ' '.join([el.text for el in txt])
                    
                try:
                    txt = soup.find_all('p', class_='doc__thought')
                    text = text + ' ' + ' '.join([el.text for el in txt])
                except:
                    pass
            except:
                text = 'No text'
            return title, announce, text

    except Exception as ex:
        print(link)
        print(ex)
        return ex

    except KeyboardInterrupt:
        return
    

data = {'link': [], 'data_or_ex': []}
current_val = 0


def start_parsing(target, parse_from_link, links, total_news, n_threads):
    
    lock = Lock()
    threads = []
    run_event = Event()
    run_event.set()

    for i in range(n_threads):
        t = Thread(target=target, args=(lock, parse_from_link, links, total_news))
        t.start()
        time.sleep(1)
        threads.append(t)

    for thread in threads:
        thread.join()

def parse(lock, parse_from_link, links, total_news):

    global data, current_val

    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Safari/605.1.15"

    while current_val < total_news:
        with lock:
            link = links[current_val]
            current_val += 1
        
        parsed = parse_from_link(link, user_agent)
        if parsed == None:
            return 0

        with lock:
            data['link'].append(link)
            data['data_or_ex'].append(parsed)
            
        time.sleep(1 + random.random())
   
    return 1


### Parsing news end ###

### Text proccessing ###

def lemm_file(df, mystem, swords):

    try:
        df['tp'] = df['data_or_ex'].map(lambda x: isinstance(x, tuple))
        df.drop(df[df.tp != True].index, inplace=True)
        df[['title', 'announce', 'text']] = pd.DataFrame(df['data_or_ex'].tolist(), index=df.index)

        df['title out'] = df['title'].copy()
        df['title origin'] = df['title'].copy()
        df['announce origin'] = df['announce'].copy()
        df['text origin'] = df['text'].copy()

        df.drop(columns=['data_or_ex', 'tp'], inplace=True)
        cols = ['title', 'announce', 'text', 'title origin', 'announce origin', 'text origin']
        for col in cols:
            df[col] = df[col].str.lower()
            df[col] = df[col].str.replace('\n', ' ')
            df[col] = df[col].str.strip()
            if col == 'title origin' or col == 'announce origin' or col == 'text origin':
                continue
            df[col] = df[col].str.replace('\W+', ' ', regex=True)
            res = []
            doc = []
            large_str = ' '.join([txt + ' splitter ' for txt in df[col]])
            large_str = mystem.lemmatize(large_str)

            for word in large_str:
                if word.strip() != '' and word not in swords:
                    if word == 'splitter':
                        res.append(doc)
                        doc = []
                    else:
                        doc.append(word)
            del large_str
            del doc
            res = [' '.join(lst) for lst in res]
            df[col] = res
        return df
    except Exception as ex:
        print(f'Error in this news delivery lemmatize')
        print(ex)
        return None
    

def add_company(df):
    for company in companies:
        reg = for_regex[company]
        reg = ' | '.join(reg)
        in_str = df['title'].str.contains(reg) | df['announce'].str.contains(reg) | df['text'].str.contains(reg)
        df[company] = in_str

    df['company'] = ''
    for index, row in df.iterrows():
        for company in companies:
            if row[company]:
                df.at[index, 'company'] += ' ' + company

    df = df.drop(companies.keys(), axis=1)
    df['company'] = df['company'].apply(lambda x: x.strip())

    return df


def add_industry(df):
    
    for industry in industries:
        reg = for_regex_industry[industry]
        reg = ' | '.join(reg)
        in_str = df['title'].str.contains(reg) | df['announce'].str.contains(reg) | df['text'].str.contains(reg)
        df[industry] = in_str

    df['industry'] = ''
    for index, row in df.iterrows():
        for industry in industries:
            if row[industry]:
                df.at[index, 'industry'] += ' ' + industry

    df = df.drop(industries, axis=1)
    df['industry'] = df['industry'].apply(lambda x: x.strip())

    return df

### Text proccessing end ###

### Machine Learning ###

def make_dataset(df):
    index = ir.SentimentIndex.load('ml/test.index', 'delta', 'bogram')
    index.get_text = lambda x: x[0]

    dataset_df = df.loc[:, ['title origin']]
    docs = []
    for ind, row in dataset_df.iterrows():
        docs.append(row['title origin'])
        

    x = []
    for doc in docs:
        x.append(index.weight(index.features(doc)))

    with open("ml/columns.txt", "r") as file:
        file_cols = file.read().split('\n')

    ml_ready_df = pd.DataFrame(columns=file_cols)

    cols = ml_ready_df.columns.tolist()

    for row in x:
        ml_ready_df = pd.concat([ml_ready_df, pd.DataFrame([row])], ignore_index=True).loc[:, cols]

    ml_ready_df = ml_ready_df.replace({True: 1, False: 0})
    ml_ready_df = ml_ready_df.fillna(0)

    return ml_ready_df

def make_estimation(ml_df):

    model = joblib.load('ml/ML_model.joblib') 
    pred = model.predict(ml_df)  
    return pred 

### Machine Learning end ###

### Data output ###

def save_db(df):

    try:
        cursor = connection.cursor()
        for index, row in df.iterrows(): 
            insert_query = f"INSERT INTO news_est_table (link, datetime, title_origin, company) VALUES ('{row['link']}', '{str(row['datetime'])}', '{row['title origin']}', '{row['company']}')"
            cursor.execute(insert_query)
        connection.commit()

    except (Exception, Error) as error:
        print("Ошибка сохранения записей PostgreSQL:", error)
        return 1
    finally:
        if connection:
            cursor.close()
            return 0


def results_out(df):
    print('Saved news: ' + str(len(df)))

### Data output end ###


### Update ML model ###

def update_ml():
    lines = check_news_db()
    if not (lines is None):
        news_df = pd.DataFrame(lines, columns=['id', 'link', 'datetime', 'title origin', 'company'])
        ml_df = add_target(news_df)
        if ml_df is None:
            print('Error in remake ML model')
            return -1
        
        create_model(ml_df)
        return 0
    else:
        print('No enought news for new ML model')
        return 1

        

def check_news_db():
    try:
        cursor = connection.cursor()
        check_query = f"SELECT count(*) FROM {table_name} WHERE company != ''"
        cursor.execute(check_query)
        record = cursor.fetchall()
        if int(record[0][0]) > 500:
            get_query = f"SELECT * FROM {table_name} WHERE company != '' ORDER BY id DESC LIMIT 500"
            cursor.execute(get_query)
            lines = cursor.fetchall()
            return lines
        else: return None

    except (Exception, Error) as error:
        print("Ошибка чтения записей PostgreSQL:", error)
        return None
    finally:
        if connection:
            cursor.close()        


def add_target(df):

    news_df = df.loc[:, ['datetime', 'title origin', 'company']]
    news_df = news_df[news_df['datetime'] != 'No time']
    news_df['datetime'] = pd.to_datetime(news_df['datetime'])
    news_df.sort_values('datetime', ignore_index=True, inplace=True)
    first_date = str(news_df.loc[0, 'datetime'].date() - pd.Timedelta(days=3))
    last_date = str(news_df.loc[len(news_df) - 1, 'datetime'].date() + pd.Timedelta(days=3))
    

    for company in companies:
            in_str = news_df['company'].str.contains(company)
            news_df[company] = in_str

    news_df = news_df.drop('company', axis=1)



    price_dict = {'company': [], 'close': [], 'datetime': []}
    price_data = pd.DataFrame.from_dict(price_dict)

    for company in companies:
        
        index = 0
        total = 100
        while index < total:
            link = f'https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{company}.xml?iss.meta=off&from={first_date}&till={last_date}&history.columns=LEGALCLOSEPRICE,TRADEDATE&start={index}'
            
            try:
                download = requests.get(link)
            except (Exception, Error) as error:
                print("Ошибка сбора котировок по компании", error)
                return None 

            decoded_content = download.content.decode('utf-8')
            root = ET.fromstring(decoded_content)
            index = int(root[1][0][0].get('INDEX'))
            index += 100
            total = int(root[1][0][0].get('TOTAL'))
            rows = root[0][0]
            for row in rows: 
                close = row.get('LEGALCLOSEPRICE')
                date = row.get('TRADEDATE')
                price_data.loc[len(data)] = [company, close, date]
            time.sleep(1 + random.random())

    
    for com in companies:
        price = price_data[price_data['company'] == com].copy()
        price = price.drop('company', axis=1)
        price['datetime'] = pd.to_datetime(price['datetime'])
        price['datetime'] = price['datetime'] + pd.DateOffset(hours=19)
        price.sort_values('datetime', ignore_index=True, inplace=True)

        news_df = pd.merge_asof(news_df, price, on='datetime', direction='backward')
        news_df.rename(columns={'close': f'{com} close prev'}, inplace=True)

        news_df = pd.merge_asof(news_df, price, on='datetime', direction='forward')
        news_df.rename(columns={'close': f'{com} close'}, inplace=True)


    res_df = pd.DataFrame()
    for company in companies:
        comp = news_df[news_df[company] == True].copy()
        comp['close'] = comp[f'{company} close'] - comp[f'{company} close prev']
        comp = comp.loc[:, ['title origin', 'close', 'datetime']]
        res_df = pd.concat([res_df, comp], axis=0)
    res_df.drop_duplicates(inplace=True)
    res_df.sort_values('datetime', ignore_index=True, inplace=True)

    return res_df



def create_model(df):

    title_origin_df = df.loc[:, ['title origin', 'close']]
    docs = []

    for ind, row in title_origin_df.iterrows():
        docs.append((row['close'], row['title origin']))

    index = ir.SentimentIndex('delta', 'bogram')
    index.get_class = lambda x: x[0]
    index.get_text = lambda x: x[1]
    index.build(docs)

    x = []
    y = []
    for doc in docs:
        x.append(index.weight(index.features(doc)))
        y.append(doc[0])

    index.save('ml2/test.index') # К ИЗМЕНЕНИЮ!
    print(f'On {supply_counter} iteration: sentiment analyzer saved')


    ml_ready_df = pd.DataFrame()
    for row in x:
        ml_ready_df = pd.concat([ml_ready_df, pd.DataFrame([row])], ignore_index=True)
    
    cols = ml_ready_df.columns.tolist()

    with open("ml2/columns.txt", "w") as file:
        # Преобразование списка в строку и запись в файл
        file.write("\n".join(cols))
        print(f'On {supply_counter} iteration: columns file saved')

    sc = pd.Series(y[:])

    ml_ready_df = pd.concat([ml_ready_df, sc], axis=1)
    ml_ready_df.rename(columns={0: 'close'}, inplace=True)

    ml_ready_df['class'] = ml_ready_df['close'] > 0
    ml_ready_df.drop(columns=['close'], inplace=True)
    ml_ready_df = ml_ready_df.replace({True: 1, False: 0})
    ml_ready_df = ml_ready_df.fillna(0)
    
    X_train = ml_ready_df.drop('class', axis=1)

    y_train = ml_ready_df['class'].values

    param_grid = {'max_depth' : list(range(1,10))}


    clf_forest = RandomForestClassifier(criterion='entropy', n_jobs=-1)

    gc = GridSearchCV(estimator=clf_forest, param_grid=param_grid, cv=2, n_jobs=1, scoring='accuracy', verbose=2, refit=True)
    gc.fit(X_train, y_train)

    preds = gc.predict(X_train)

    print(f'ML model on {supply_counter} iteration builded. Accuracy: {np.mean(y_train==preds)}')

    estimator = gc.best_estimator_
    joblib.dump(estimator, "ml2/ML_model.joblib")


### Update ML model end###


main()