## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports / Logging Setup    --
#···············································································
from attrdict import AttrDict
import datasets
from datasets import load_dataset
from eval import evaluate_model
import logging
import os
from packaging import version
from preprocess import preprocess_fn
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


<<<<<<< HEAD

=======
>>>>>>> 5b178064d19bafbf85e6e992976794c99bd823f6
# 
## ─────────────────────────────────────────────────────────────────────────────




## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     Main     --
#···············································································
def main():
    # load config
    args = toml.load('config.toml')
    _init = AttrDict(args['init'][0])
    _train = AttrDict(args['train'][0]) 

    logger.info(f'Starting script with arguments: {args}')

    wandb.init(project=_init.wandb_project, config=args)

    os.makedirs(_init.output_dir, exist_ok=True)


    raw_datasets, tokenizer, model, column_names = utils.get_tokenizer_and_model(
            lang_pairs=_init.lang_pairs,
            split=_init.split,
            output_dir=_init.output_dir,
            model_checkpoint=_init.model_checkpoint,
            use_cache=False,
            )


    train_data, eval_data = utils.get_datasets(
            output_dir=_init.output_dir,
            raw_datasets=raw_datasets,
            split=_init.split,
            lang_pairs=_init.lang_pairs,
            preprocess_fn=preprocess_fn,
            max_seq_length=_init.max_seq_length,
            tokenizer=tokenizer,
            batched=True,
            num_proc=_init.preprocessing_num_workers,
            column_names=column_names,
            load_from_cache_file=True,
            all_langs=_init.all_langs,
            )

    data_collator = transformers.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_dataloader, eval_dataloaders = utils.get_dataloaders(
            data_collator=data_collator,
            train_dataset=train_data,
            eval_data=eval_data,
            batch_size=_train.batch_size
            )

<<<<<<< HEAD
    optimizer = torch.optim.Adam(model.parameters(), lr=_train.learning_rate, weight_decay=_train.weight_decay)

    num_update_steps_per_epoch = len(train_dataloader)
    if max_train_steps == 0:
        max_train_steps = int(num_train_epochs) * num_update_steps_per_epoch
    else:
        num_train_epochs = math.ceil(int(max_train_steps) / num_update_steps_per_epoch)
=======

    optimizer, lr_scheduler, max_train_steps, num_train_epochs = utils.get_optim_and_lr_scheduler(
            output_dir=_init.output_dir,
            model=model,
            learning_rate=_train.learning_rate,
            max_train_steps=_train.max_train_steps,
            num_train_epochs=_train.num_train_epochs,
            train_dataloader=train_dataloader,
            lr_scheduler_type=_train.lr_scheduler_type,
            accum_iter=_train.accum_iter,
            num_warmup_steps=_train.num_warmup_steps,
            weight_decay=_train.weight_decay,
            )
>>>>>>> 5b178064d19bafbf85e6e992976794c99bd823f6

    lr_scheduler = transformers.get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps // accum_iter,
    )

   

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
    logger.info(f"  Num Epochs = {_train.num_train_epochs}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    progress_bar = tqdm(range(max_train_steps))

    # Log a pre-processed training example to make sure the pre-processing does not have bugs in it
    # and we do not input garbage to our model
    batch = next(iter(train_dataloader))

    _labs= batch['labels']
    _labs[_labs == -100] = tokenizer.pad_token_id
    logger.info("Look at the data that we input into the model, check that it looks as expected: ")
    for index in random.sample(range(len(batch)), 2):
        logger.info(f"Decoded input_ids: {tokenizer.decode(batch['input_ids'][index])}")
        logger.info(f"Decoded labels: {tokenizer.decode(batch['labels'][index])}")
        logger.info("\n")
    _labs[_labs == tokenizer.pad_token_id] = -100 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    global_step = 0
    model = model.to(device)

    for epoch in range(_train.num_train_epochs):
        model.train()

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            labels[labels == tokenizer.pad_token_id] = -100

            with torch.set_grad_enabled(True):

                output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = output['loss']

                loss = loss /_train.accum_iter
                
                loss.backward()

                if ((batch_idx + 1) % _train.accum_iter == 0) or (batch_idx + 1 == len(train_dataloader)):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

        
            wandb.log(
                        {
                            'train_loss' : loss,
                            'learning_rate' : optimizer.param_groups[0]['lr'],
                            'epoch' : epoch,
                            },
                        step=global_step,
                        )

            if global_step % _train.eval_every_steps == 0 or global_step == _train.max_train_steps:
                for lang_pair, eval_dataloader in zip(_init.lang_pairs, eval_dataloaders):
                    eval_results, last_input_ids, last_decoded_preds, last_decoded_labels = evaluate_model(
                            model=model,
                            dataloader=eval_dataloader,
                            tokenizer=tokenizer,
                            device=device,
                            max_seq_length=_init.max_seq_length,
                            beam_size=_train.beam_size,
                        )

                    wandb.log(
                             {
                                 f'{lang_pair}_eval/bleu' : eval_results['bleu'],
                                 f'{lang_pair}_eval/generation_length': eval_results['generation_length'],
                                 },
                             step=global_step,
                             )
                    logger.info("Generation example:")
                    random_index = random.randint(0, len(last_input_ids) - 1)
                    logger.info(f"Input sentence: {tokenizer.decode(last_input_ids[random_index], skip_special_tokens=True)}")
                    logger.info(f"Generated sentence: {last_decoded_preds[random_index]}")
                    logger.info(f"Reference sentence: {last_decoded_labels[random_index][0]}")
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
        raise RuntimeError("This script requires Datasets 1.18.0 or higher. Please update via pip install -U datasets.")

    main()








#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


