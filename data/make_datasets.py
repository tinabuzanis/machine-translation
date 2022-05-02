import datasets
import pandas as pd

fr = open('fr.txt').readlines()
ru = open('ru.txt').readlines()

fr  = [sentence.strip() for sentence in fr]
ru =  [sentence.strip() for sentence in ru]


def to_dataset(data):
    df = pd.DataFrame(data, columns=['tn'])
    df['ids'] = df.index
    df['translation'] = df['tn']
    df = df.drop('tn', axis=1)

    return datasets.Dataset.from_pandas(df)

ru = ru[:10]
fr = fr[:10]

t =  [{'fr' : fr_el, 'ru' : ru_el} for fr_el, ru_el in zip(fr,  ru)]


m = pd.DataFrame(t)

datasets.Dataset.from_pands(m)



