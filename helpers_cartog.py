#I never got this to work - the trainer was not finding the 'example_id' values
# python run_cartog.py --do_train --task qa_adv --output_dir ./trained_model_adv/ --num_train_epochs 1

import numpy as np
import collections
import torch
from collections import defaultdict, OrderedDict
from transformers import Trainer, EvalPrediction, TrainerCallback
from transformers.trainer_utils import PredictionOutput
from typing import Tuple
from tqdm.auto import tqdm
import csv
QA_MAX_ANSWER_LENGTH = 30
from transformers import default_data_collator
 
#this function written by chatgpt to get the trainer to import this extra key
def qa_data_collator(features):
    # default collator builds tensors for input_ids, attention_mask, etc
    batch = default_data_collator(features)

    if "example_id" in features[0]:
        print("FOUND FEATURES")
        batch["example_id"] = torch.tensor(
            [int(f["example_id"]) for f in features],
            dtype=torch.long
        )

    return batch
'''
def qa_data_collator(features):
    batch = default_data_collator(features)
    if "example_id" in features[0]:
        print("FOUND FEATURES")
        batch["example_id"] = torch.tensor([f["example_id"] for f in features], dtype=torch.long)
        print("COLLATOR:  batch[example_id] = ", batch["example_id"])
    return batch
'''
# This function computes sentence-classification accuracy.
# Functions with signatures like this one work as the "compute_metrics" argument of transformers.Trainer.
def compute_accuracy(eval_preds: EvalPrediction):
    return {
        'accuracy': (np.argmax(
            eval_preds.predictions,
            axis=1) == eval_preds.label_ids).astype(
            np.float32).mean().item()
    }


# This function preprocesses a question answering dataset, tokenizing the question and context text
# and finding the right offsets for the answer spans in the tokenized context (to use as labels).
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def prepare_train_dataset_qa(examples, tokenizer, max_seq_length=None):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    
    tokenized_examples["example_id"] = [] #I added this
    
    #id2int = {eid: i for i, eid in enumerate(examples["id"])}#added by chatgpt in debugging - strings were not working
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        ex_id = examples["id"][sample_index]
        #print("In train_prep: ex_id = ", ex_id) #string of letters and numbers
        # map onto an integer instead (this next line is from chatgpt to map onto integers for ID purposes)
        #ex_id_int = id2int[ex_id]                # convert to integer
        #tokenized_examples["example_id"].append(ex_id_int)
        ############
        tokenized_examples["example_id"].append(ex_id) #I added this
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    assert len(tokenized_examples["input_ids"]) == len(tokenized_examples["example_id"])
    assert len(tokenized_examples["input_ids"]) == len(tokenized_examples["start_positions"])
    assert len(tokenized_examples["input_ids"]) == len(tokenized_examples["end_positions"])
    return tokenized_examples


def prepare_validation_dataset_qa(examples, tokenizer):
    questions = [q.lstrip() for q in examples["question"]]
    max_seq_length = tokenizer.model_max_length
    tokenized_examples = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=min(max_seq_length // 2, 128),
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


# This function uses start and end position scores predicted by a question answering model to
# select and extract the predicted answer span from the context.
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py
def postprocess_qa_predictions(examples,
                               features,
                               predictions: Tuple[np.ndarray, np.ndarray],
                               n_best_size: int = 20):
    if len(predictions) != 2:
        raise ValueError(
            "`predictions` should be a tuple with two elements (start_logits, end_logits).")
    all_start_logits, all_end_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features.")

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[
            example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits
            # to span of texts in the original context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                            -1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[
                          -1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or \
                            end_index - start_index + 1 > QA_MAX_ANSWER_LENGTH:
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0],
                                        offset_mapping[end_index][1]),
                            "score": start_logits[start_index] +
                                     end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"],
                             reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        # In the very rare edge case we have not a single non-null prediction,
        # we create a fake prediction to avoid failure.
        if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0,
                                   "end_logit": 0.0, "score": 0.0})

        all_predictions[example["id"]] = predictions[0]["text"]
    return all_predictions
#code I added:
class EpochEndCallback(TrainerCallback):
    def __init__(self, trainer_instance,dynamics_output_dir = "cartog"):
        super().__init__()
        self.dynamics_output_file = "cartog/training_dynamics.csv"
        #self.epoch_data = []
        self.counter = 0
        self.trainer = trainer_instance
        #clear the output file by writing a header to it (use encoding to avoid unicode errors)
        with open(self.dynamics_output_file, 'w', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['example_id', 'predicted_answer', 'true_answer',
                             'joint_prob', 'is_correct'])     

    def on_epoch_end(self, args, state, control, **kwargs):
        print("End of epoch*******************",flush  = True)

        """Save dynamics at end of each epoch
            in format example_id, predicted, true_answer, p(y|x), is_correct"""
         
        with open(self.dynamics_output_file, 'a', newline='', encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            count = 1
            empty_count = 0
            for d in self.trainer.epoch_data:
                ex_id = d["example_id"]
                pred = d["predicted_answer"]
                if len(pred) <1: empty_count += 1
                true = d["true_answer"]
                prob = d["gold_joint_prob"]
                correctness = d["is_correct"]
                writer.writerow([ex_id,pred.replace(',',''),
                                 true.replace(',',''), prob, correctness])
                count += 1
                if self.counter < 5:
                    print(ex_id,pred,true,prob,correctness,flush=True)
                    self.counter += 1
        print("Empty predictions in this epoch: ", empty_count)
                 
        self.trainer.epoch_data = []# Reset for next epoch
        print("Tokenized example counts in this epoch: ", count)
        return control
# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/trainer_qa.py
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None,dynamics_output_dir = "cartog",**kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.epoch_data = []
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        This function written by Claude.ai
        To capture outputs after each epoch and write them to a file
        """
         
        # Extract example IDs before they're potentially removed
        example_ids = inputs.get('example_id', None)
        print("*** example_ids in compute_loss:", example_ids )
        #print("inputs*****",inputs)
        #print("inputs['example_id']" , inputs['example_id'])
        input_ids_for_decode = inputs['input_ids'].clone()
        
        # Call parent's compute_loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        #distrib of probabilities. Shape: [batch_size, sequence_length]
        start_probs = torch.softmax(outputs.start_logits, dim=-1)
        end_probs = torch.softmax(outputs.end_logits, dim=-1)

        #predicted start/end positions. shape: [batch_size]
        #chooses best result by individually maximizing logits
        pred_start = torch.argmax(outputs.start_logits, dim=1)
        pred_end = torch.argmax(outputs.end_logits, dim=1)
        true_start = inputs['start_positions']
        true_end = inputs['end_positions']
        # Process each example in the batch
        for i in range(input_ids_for_decode.shape[0]):
            
            # Decode predicted answer
            ps, pe = pred_start[i].item(), pred_end[i].item()
            if ps <= pe and pe < input_ids_for_decode.shape[1]:
                pred_answer = self.tokenizer.decode(
                    input_ids_for_decode[i, ps:pe + 1], 
                    skip_special_tokens=True
                )
            else:
                pred_answer = ""
            # Decode ground truth answer
            ts, te = true_start[i].item(), true_end[i].item()
            
            if ts <= te and te < input_ids_for_decode.shape[1]:
                true_answer = self.tokenizer.decode(
                    input_ids_for_decode[i, ts:te + 1], 
                    skip_special_tokens=True
                )
            else:
                true_answer = ""

            # Calculate confidence: probability assigned to gold label
            # For QA, this is the joint probability of start and end positions
            gold_start_prob = start_probs[i, ts].item() if ts < start_probs.shape[1] else 0.0
            gold_end_prob = end_probs[i, te].item() if te < end_probs.shape[1] else 0.0
            joint_prob = gold_start_prob * gold_end_prob  # Joint probability
            
            # Check if prediction is correct
            is_correct = (ps == ts and pe == te)
                
            # Store dynamics
            self.epoch_data.append({
                'example_id': example_ids[i].item() if example_ids is not None else None,
                'predicted_answer': pred_answer,
                'true_answer': true_answer,
                'gold_joint_prob': joint_prob,
                'is_correct': int(is_correct)
            })
        return (loss, outputs) if return_outputs else loss
    

    
    #the following methods are from the original template
    def evaluate(self,
                 eval_dataset=None,  # denotes the dataset after mapping
                 eval_examples=None,  # denotes the raw dataset
                 ignore_keys=None,  # keys to be ignored in dataset
                 metric_key_prefix: str = "eval"
                 ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # compute the raw predictions (start_logits and end_logits)
            output = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # post process the raw predictions to get the final prediction
            # (from start_logits, end_logits to an answer string)
            eval_preds = postprocess_qa_predictions(eval_examples,
                                                    eval_dataset,
                                                    output.predictions)
            formatted_predictions = [{"id": k, "prediction_text": v}
                                     for k, v in eval_preds.items()]
            references = [{"id": ex["id"], "answers": ex['answers']}
                          for ex in eval_examples]

            # compute the metrics according to the predictions and references
            metrics = self.compute_metrics(
                EvalPrediction(predictions=formatted_predictions,
                               label_ids=references)
            )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state,
                                                         self.control, metrics)
        return metrics
