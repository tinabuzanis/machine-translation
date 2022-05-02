import datasets
from datasets import load_metric
from preprocess import postprocess_text
import torch
from tqdm import tqdm

bleu = load_metric('sacrebleu')

def evaluate_model(
        model,
        dataloader,
        *,
        tokenizer,
        device,
        max_seq_length,
        beam_size,
        ):
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


