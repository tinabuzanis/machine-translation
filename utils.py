import datasets
from datasets import load_dataset, concatenate_datasets
import datasets
from functools import partial
import logging
import math
import pickle
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
import os

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def get_tokenizer_and_model(
        lang_pairs, 
        split,
        model_checkpoint,
        all_langs,
        use_cache=False,
        ):
    tokenizer = transformers.MT5Tokenizer.from_pretrained(model_checkpoint, truncation=True)
    model = transformers.MT5ForConditionalGeneration.from_pretrained(model_checkpoint, use_cache=use_cache)
    raw_datasets =  []
    column_names = ""

    if lang_pairs == 'fr-en':
        fr_en_train = datasets.load_dataset('wmt14', 'fr-en', split='train[:1000000]')
        fr_en_eval = datasets.load_dataset('wmt14', 'fr-en', split='test[:3000]') 
        fr_en = datasets.DatasetDict({'train' : fr_en_train, 'test' : fr_en_eval})
        raw_datasets = [fr_en]
        column_names = ['translation']

    if lang_pairs[0] == 'fr-ru':
        logger.info('Creating fr-ru dataset... This may take a moment')

        with open('fr_ru_train.pkl', 'rb') as f:
            train_dict = pickle.load(f)

        with open('fr_ru_test.pkl', 'rb') as f:
            test_dict = pickle.load(f)
        train_dataset = datasets.Dataset.from_dict(train_dict)
        test_dataset = datasets.Dataset.from_dict(test_dict)

        fr_ru_dataset = datasets.DatasetDict({'train' : train_dataset, 'test' : test_dataset})
        column_names = ['translation']
        return [fr_ru_dataset], tokenizer, model, column_names


    elif all_langs==True:
        fr_en_train = datasets.load_dataset('wmt14', 'fr-en', split='train[:148000]')
        fr_en_eval = datasets.load_dataset('wmt14', 'fr-en', split='test[:3000]') 
        fr_en = datasets.DatasetDict({'train' : fr_en_train, 'test' : fr_en_eval})
        ru_en = datasets.load_dataset('wmt14', 'ru-en')
        raw_datasets = [fr_en, ru_en]
        column_names=['translation']


    else:
        raw_datasets = []
        for lp in lang_pairs:
            if split == 'None':
                raw_datasets.append(load_dataset('wmt14', lp))
            else:
                raw_datasets.append(load_dataset('wmt14', lp, split=split))


        if split == 'None':
            column_names = raw_datasets[0]['train'].column_names
        else:
            column_names = ['translation']

    return raw_datasets, tokenizer, model, column_names




def get_datasets(
        raw_datasets,
        split,
        lang_pairs,
        preprocess_fn,
        max_seq_length,
        tokenizer,
        batched,
        num_proc,
        column_names,
        load_from_cache_file):

    train_langs = []
    test_langs = []
    model_outputs = []
    for i, lang_pair in enumerate(lang_pairs):
        src, tgt = lang_pair.split('-')
        preprocess_fn_wrapped = partial(
                preprocess_fn,
                source_lang=src,
                target_lang=tgt,
                max_seq_length=max_seq_length,
                tokenizer=tokenizer,
                )
        model_output = raw_datasets[i].map(
                preprocess_fn_wrapped,
                batched=batched,
                num_proc=num_proc,
                remove_columns=column_names,
                load_from_cache_file=load_from_cache_file,
                desc=f'Tokenizing {lang_pair} dataset',
                )
        if split == 'None':
            train_langs.append(model_output['train'])
            test_langs.append(model_output['test'])
        else:
            model_output = model_output.train_test_split(test_size=0.2)
            model_outputs.append(model_output)

    if split == 'None':
        train_dataset = concatenate_datasets([*train_langs])

        eval_data = test_langs
    else:
        # concat every training sub-section for each lang
        train_data_to_concat = []
        eval_data= []
        for i, _ in enumerate(lang_pairs):
            train_data_to_concat.append(model_outputs[i]['train'])
            eval_data.append(model_outputs[i]['test'])

        train_dataset = concatenate_datasets([*train_data_to_concat])

# Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        logger.info(f"Decoded input_ids: {tokenizer.decode(train_dataset[index]['input_ids'])}")
        logger.info(f"Decoded labels: {tokenizer.decode(train_dataset[index]['labels'])}")
        print('LEN TEST LANGS', len(eval_data))
        logger.info("\n")

    return train_dataset, eval_data


def get_dataloaders(
        data_collator,
        train_dataset,
        eval_data,
        batch_size):

    

    train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=8,
            batch_size=batch_size,
            pin_memory=True
            )

    eval_dataloaders = []
    for ed in eval_data:
        eval_dataloaders.append(
                DataLoader(
                    ed,
                    shuffle=False,
                    collate_fn=data_collator,
                    batch_size=batch_size,
                    pin_memory=True
                    )
                )
    return train_dataloader, eval_dataloaders



