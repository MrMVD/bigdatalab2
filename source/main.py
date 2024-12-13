from pyspark import RDD
from pyspark.sql import SparkSession
import nltk
from nltk.corpus import stopwords               # Для удаления стоп-слов
from nltk.stem.snowball import SnowballStemmer  # Стеммер для русского языка
import re
import os
import sys

# Чтобы не засирать системыне переменные
os.environ['PYSPARK_PYTHON'] = sys.executable

# Инициализация Spark
spark = SparkSession\
        .builder\
        .appName("RussianTextAnalysis")\
        .getOrCreate()

# Загрузка зависимостей NLTK
nltk.download("stopwords")

# Загрузка файла как RDD 
text_rdd = spark.read.text("../text.txt" ).rdd.map(lambda r: r[0])

# стоп-слов для русского языка
stop_words = set(stopwords.words("russian"))

# Очистка RDD
clear_rdd =(text_rdd                                                     # Исходный текст
    .flatMap(lambda line: re.sub(r'[^\w\s]', '', line).lower().split())  # Очистка и празделение
    .filter(lambda word: word not in stop_words))                        # Удаление стоп-слов
                         
# Инициализация стеммера для русского языка
stemmer = SnowballStemmer("russian")

# Стеминг (поиск основы)
stemmed_clear_rdd = clear_rdd.map(lambda word: stemmer.stem(word))

def PrintWordFrequency(rdd:RDD):
    # Подсчёт пар с одинаковым словом
    counted_word_rdd = rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
    # Функция вывода
    def InerPrint(rdd:RDD, ascending: bool):
        words =(rdd
            .sortBy(lambda x: x[1], ascending=ascending)
            .take(50))
        print("Топ-50 самых " +("редких" if ascending else "частых")+ " слов:")
        for word, count in words:
            print(f"{word}: {count}")
    
    InerPrint(counted_word_rdd,False) # вывод частых слов
    InerPrint(counted_word_rdd,True)  # вывод редких слов

print("Обычные слова")
PrintWordFrequency(clear_rdd)
print("Основы слов")
PrintWordFrequency(stemmed_clear_rdd)