code_to_lang = {
        'fr' : 'French',
        'en' : 'English', 
        'ru' : 'Russian'
        }
def set_prefix(ex):
    src, tgt = ex.keys()
    src = code_to_lang[src]
    tgt = code_to_lang[tgt]
    return f"translate from {src} to {tgt}: "

def preprocess_fn(
        examples,
        source_lang,
        target_lang,
        max_seq_length,
        tokenizer,
        ):

    inputs = []
    targets = []

    inputs.extend([set_prefix(ex) + ex[source_lang] for ex in examples['translation']])
    targets.extend([ex[target_lang] for ex in examples['translation']])

    inputs.extend([set_prefix(ex) + ex[target_lang] for ex in examples['translation']])
    targets.extend([ex[source_lang] for ex in examples['translation']])

    model_inputs=tokenizer(inputs, max_length=max_seq_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_seq_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']

    return model_inputs

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels
