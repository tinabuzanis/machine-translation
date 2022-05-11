

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports     --
#···············································································
from attrdict import AttrDict
import datasets
from datasets import load_dataset
from functools import partial
import logging
import math
import os
from packaging import version
import random
import toml
import torch
import transformers
from tqdm.auto import tqdm
import utils
import wandb


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_warning()

bleu = datasets.load_metric("sacrebleu")


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

#TODO:
# make optimizer a config arg
# use largest mt5 model after doing these timizations? - XL
# make multilingual model filtered + try w/ adafactor / Adam 
# copy config from MarianMT?
# train bilingual models in one direction (first) and evaluate. is our BLEU better?
# save tokenizer for each model (vocab, config, etc)
# save each model 
def preprocess_fn(examples, tokenizer, max_seq_length):
    inputs, targets = [], []

    inputs.extend([f'translate from English to Russian: ' + ex['en'] for ex in examples['translation']])
    targets.extend([ex['ru'] for ex in examples['translation']])

    model_inputs = tokenizer(inputs, max_length=max_seq_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_seq_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']

    return  model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def evaluate_model(model, dataloader, *, tokenizer, device, max_seq_length, beam_size):

    n_generated_tokens = 0
    model.eval()
    for batch in tqdm(dataloader, desc='Evaluation'):
        with torch.inference_mode():
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            generated_ids = model.generate(
                    input_ids=input_ids,
                    num_beams=beam_size,
                    attention_mask=attention_mask
                    )

            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            decoded_labels = [tokenizer.decode(l, skip_special_tokens=True, clean_up_tokenization_spaces=True) for l in labels]

            for pred in decoded_preds:
                n_generated_tokens += len(tokenizer(pred)['input_ids'])

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            bleu.add_batch(predictions=decoded_preds, references=decoded_labels)
        
    model.train()
    eval_metric = bleu.compute()
    evaluation_results = {
            'bleu' : eval_metric['score'],
            'generation_length' : n_generated_tokens / len(dataloader.dataset)
            }

    return evaluation_results, input_ids, decoded_preds, decoded_labels


def main():
    args = toml.load('config.toml')
    _init = AttrDict(args["init"][0])
    _train = AttrDict(args["train"][0])
    wandb.init(project=_init.wandb_project, config=args)

    os.makedirs(_init.output_dir, exist_ok=True)


    column_names = ['translation']
    # EN - RU
    en_ru_tokenizer = transformers.MT5Tokenizer.from_pretrained(_init.en_ru)
    en_ru_model = transformers.MT5ForConditionalGeneration.from_pretrained(_init.en_ru)

    ds1 = datasets.load_dataset('opus_books', 'en-ru')
    ds2 = datasets.load_dataset('opus_wikipedia', 'en-ru')

    en_ru_dataset = datasets.concatenate_datasets([ds1['train'], ds2['train']])
    en_ru_dataset = en_ru_dataset.train_test_split(test_size=3000)

    preprocess_fn_wrapped = partial(preprocess_fn, max_seq_length=_init.max_seq_length, tokenizer=en_ru_tokenizer)
    model_output = en_ru_dataset.map(preprocess_fn_wrapped, batched=True, num_proc=_init.preprocessing_num_workers, remove_columns=en_ru_dataset['train'].column_names, load_from_cache_file=True, desc='Tokenizing EN-RU')

    train_dataset, test_dataset = model_output['train'], model_output['test']


    # for index in random.sample(range(len(train_dataset)), 2):
        # logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        # logger.info(f"Decoded input_ids: {en_ru_tokenizer.decode(train_dataset[index]['input_ids'])}")
        # logger.info(f"Decoded labels: {en_ru_tokenizer.decode(train_dataset[index]['labels'])}")
        # print('LEN TEST LANGS', len(test_dataset))
        # logger.info("\n")


    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=en_ru_tokenizer, model=en_ru_model)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, num_workers=32, batch_size=_train.batch_size, pin_memory=True)
    eval_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, collate_fn=data_collator, batch_size=_train.batch_size, pin_memory=True)

    optimizer = transformers.optimization.Adafactor(en_ru_model.parameters(), lr=_train.learning_rate, relative_step=False, scale_parameter=False)


    num_update_steps_per_epoch = len(train_dataloader)
    if _train.max_train_steps == 0:
        max_train_steps = int(_train.num_train_epochs) * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(int(_train.max_train_steps) / num_update_steps_per_epoch)

    lr_scheduler = transformers.get_scheduler(
        name=_train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=_train.num_warmup_steps,
        num_training_steps=max_train_steps // _train.accum_iter,
    )

    logger.info("***** Running training *****")


    # Log a pre-processed training example to make sure the pre-processing does not have bugs in it
    # and we do not input garbage to our model
    batch = next(iter(train_dataloader))

    # _labs = batch["labels"]
    # _labs[_labs == -100] = en_ru_tokenizer.pad_token_id
    # logger.info(
        # "Look at the data that we input into the model, check that it looks as expected: "
    # )
    # for index in random.sample(range(len(batch)), 2):
        # logger.info(f"Decoded input_ids: {en_ru_tokenizer.decode(batch['input_ids'][index])}")
        # logger.info(f"Decoded labels: {en_ru_tokenizer.decode(batch['labels'][index])}")
        # logger.info("\n")
    # _labs[_labs == en_ru_tokenizer.pad_token_id] = -100

    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {_train.num_train_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps))


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    global_step = 0
    model = en_ru_model.to(device)

    for epoch in range(_train.num_train_epochs):
        model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels[labels == en_ru_tokenizer.pad_token_id] = -100

            with torch.set_grad_enabled(True):

                output = model(
                    input_ids=input_ids, labels=labels, attention_mask=attention_mask
                )
                loss = output["loss"]

                loss = loss / _train.accum_iter

                loss.backward()

                if ((batch_idx + 1) % _train.accum_iter == 0) or (
                    batch_idx + 1 == len(train_dataloader)
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            wandb.log(
                {
                    "train_loss": loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=global_step,
            )

            if (
                global_step % _train.eval_every_steps == 0
                or global_step == _train.max_train_steps
                or global_step == 1000
                or global_step == 5000
            ):
                # for lang_pair, eval_dataloader in zip(
                    # _init.lang_pairs, eval_dataloaders
                # ):
                (
                    eval_results,
                    last_input_ids,
                    last_decoded_preds,
                    last_decoded_labels,
                ) = evaluate_model(
                    model=model,
                    dataloader=eval_dataloader,
                    tokenizer=en_ru_tokenizer,
                    device=device,
                    max_seq_length=_init.max_seq_length,
                    beam_size=_train.beam_size,
                )

                wandb.log(
                    {
                        f"[ru-en]_eval/bleu": eval_results["bleu"],
                        f"[ru-en]_eval/generation_length": eval_results[
                            "generation_length"
                        ],
                    },
                    step=global_step,
                )
                logger.info("Generation example:")
                random_index = random.randint(0, len(last_input_ids) - 1)
                logger.info(
                    f"Input sentence: {en_ru_tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}"
                )
                logger.info(
                    f"Generated sentence: {last_decoded_preds[random_index]}"
                )
                logger.info(
                    f"Reference sentence: {last_decoded_labels[random_index][0]}"
                )
                logger.info("Saving model checkpoint to %s", _init.output_dir)
                model.save_pretrained(_init.output_dir)
            # wandb.save(os.path.join(_init.output_dir, "*"))
        # YOUR CODE ENDS HERE logger.info("Saving final model checkpoint to %s", args.output_dir)
        model.save_pretrained(_init.output_dir)

        logger.info("Uploading tokenizer, model and config to wandb")
        wandb.save(os.path.join(_init.output_dir, "*"))

        logger.info(f"Script finished succesfully, model saved in {_init.output_dir}")


if __name__ == "__main__":
    if version.parse(datasets.__version__) < version.parse("1.18.0"):
        raise RuntimeError(
            "This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets."
        )

    main()










