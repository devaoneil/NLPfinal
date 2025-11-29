import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers_cartog import prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy,\
    EpochEndCallback, qa_data_collator
import os
import json
import warnings
import shutil
warnings.filterwarnings("ignore", message=".*torch.load.*")
''' NOTE: pass  --learning_rate 2e-5 at runtime; otherwise it predicts the same answer for everything'''

NUM_PREPROCESSING_WORKERS = 2
def flatten_nested_dataset(nested_dataset):
    """
    Transform nested dataset structure to flat structure compatible with SQuAD format.
    This function was written by Claude.ai
    Args:
        nested_dataset: List of dicts with structure:
            [{"title": str, "paragraphs": [{"context": str, "qas": [{"question":
            str, "answers": [...], "id": str}]}]}]
    Returns:
        Dict with flat structure: {"question": [...], "context": [...], "answers": [...], "id": [...]}
    """
    flattened = {
        "question": [],
        "context": [],
        "answers": [],
        "id": []
    }
    
    for article in nested_dataset:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                flattened["question"].append(qa["question"])
                flattened["context"].append(context)
                
                # Transform answers from list of dicts to dict of lists (SQuAD format)
                original_answers = qa["answers"]
                squad_format_answers = {
                    "text": [],
                    "answer_start": []
                }
                for ans in original_answers:
                    squad_format_answers["text"].append(ans["text"])
                    squad_format_answers["answer_start"].append(ans["answer_start"])
                
                flattened["answers"].append(squad_format_answers)
                flattened["id"].append(qa["id"])
    return flattened
#Preprocess data if not using default Huggingface
# I added this function to convert a .json file from list to flattened dict
def save_as_squad(raw_json, tag):
    out_json = 'formatted_' + tag + '.json'
    with open(raw_json, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)
        flat_dict = flatten_nested_dataset(raw_dataset)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(flat_dict, f, indent = 2)



def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa','qa_adv'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')

    training_args, args = argp.parse_args_into_dataclasses()

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset is not None and (args.dataset.endswith('.json') or args.dataset.endswith('.jsonl')):
        dataset_id = None
        # Load from local json/jsonl file #I added this next line
        #if training_args.do_train:
        #    save_as_squad(args.dataset,'train') #reformats json and renames the json file
        #    dataset = datasets.load_dataset('json', data_files='formatted_train.json')
        #if training_args.do_eval:
        #    save_as_squad(args.dataset,'eval') #reformats json and renames the json file
        #    dataset = datasets.load_dataset('json', data_files='formatted_eval.json')
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else: #using --dataset flag; I'm doing it this way: specify 'qa' or 'qa_adv' for training
        default_datasets = {'qa_adv': ('adversarialQA',),'qa': ('squad',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        eval_split = 'validation'
        # Load the raw data
        if args.task == 'qa_adv':
            dataset =  datasets.load_dataset("UCLNLP/adversarial_qa", "adversarialQA",trust_remote_code=True)
        else:   #task = 'qa' (regular squad) - use this if you want to train on squad
            # Clear the squad cache
           # cache_path = os.path.expanduser("~/.cache/huggingface/datasets/squad")
           # if os.path.exists(cache_path):
           #     print(f"Removing cache at {cache_path}")
            #    shutil.rmtree(cache_path)
    
            dataset = datasets.load_dataset("squad", trust_remote_code=True)
    
    ##########DEBUG - check if falling back to squad *****
    # After loading dataset
    sample_example = dataset[eval_split][0]#grabs 'validation' set if in .do_eval
    print(f"Sample question: {sample_example.get('question', 'NO QUESTION FIELD')}")
    print(f"Sample context preview: {sample_example.get('context', 'NO CONTEXT')[:]}...")
    ###########################################################
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'qa_adv': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs,trust_remote_code=True)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa' or args.task == 'qa_adv':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa' or args.task == 'qa_adv':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    
    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,data_collator=qa_data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:

        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback
        #I added the following custom callback:
        epoch_end_callback = EpochEndCallback(trainer_instance = trainer)
        trainer.add_callback(epoch_end_callback)
        trainer.train()
        trainer.save_model()
    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa' or args.task == 'qa_adv':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()
