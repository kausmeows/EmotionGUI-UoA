import dataclasses
import collections
import collections.abc
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
import audeer
import audmetric
import audiofile
import typing
import transformers
from datasets import load_dataset, Audio, Dataset
from transformers import Wav2Vec2Processor, EvalPrediction
from transformers.trainer import Trainer, TrainingArguments
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from transformers.integrations import TensorBoardCallback

# Training and evaluation parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
metrics = {
    'PCC': audmetric.pearson_cc,
    'CCC': audmetric.concordance_cc,
    # 'MSE': audmetric.mean_squared_error,
    # 'MAE': audmetric.mean_absolute_error,

}
log_root = audeer.mkdir('log')
model_root = audeer.mkdir('model')
SAMPLING_RATE = 16000

# ----------------- Training -----------------


class ConcordanceCorCoeff(torch.nn.Module):
    r"""
    Criterion class for the training loss function
    """
    def __init__(self):
        super().__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(self, prediction, ground_truth):
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        var_gt = self.var(ground_truth, 0)
        var_pred = self.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (self.sqrt(self.sum(v_pred ** 2))
                                         * self.sqrt(self.sum(v_gt ** 2)))

        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2*cor*sd_gt*sd_pred
        denominator = var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/denominator

        return 1-ccc


class Wav2Vec2ClassificationHead(nn.Module):
    r"""
    Head for wav2vec2 classification task.
    Uses a projection layer to project the hidden states to the number of classes.
    Used for finetuning the wav2vec2 model on a downstream classification task.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@dataclasses.dataclass
class SpeechClassifierOutput(transformers.file_utils.ModelOutput):
    r"""
    Output type of models that perform speech classification.
    """
    loss: typing.Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    r"""
    Wav2Vec2 model with a classification head on top of the transformer model
    """

    def __init__(self, config):

        super().__init__(config)

        self.num_labels = config.num_labels
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.head = Wav2Vec2ClassificationHead(config)
        self.init_weights()

    def freeze_feature_extractor(self):
        r"""
        Freezes the CNN layers of the model
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def pooling(
            self,
            hidden_states,
            attention_mask,
    ):
        r"""
        Pooling method for the hidden states
        """
        if attention_mask is None:   # For evaluation with batch_size==1
            outputs = torch.mean(hidden_states, dim=1)
        else:
            attention_mask = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            hidden_states = hidden_states * torch.reshape(
                attention_mask,
                (-1, attention_mask.shape[-1], 1),
            )
            outputs = torch.sum(hidden_states, dim=1)
            attention_sum = torch.sum(attention_mask, dim=1)
            outputs = outputs / torch.reshape(attention_sum, (-1, 1))
        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states_framewise = outputs.last_hidden_state
        hidden_states = self.pooling(
            hidden_states_framewise,
            attention_mask,
        )
        logits = self.head(hidden_states)
        loss = None

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
        )


class DataCollatorCTCWithPadding:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
    ):
        r"""Data collator that will dynamically pad the inputs received."""
        self.processor = processor

    def __call__(
        self,
        data: typing.List[typing.Dict[str, typing.Union[typing.List[int], torch.Tensor]]],
    ) -> typing.Dict[str, torch.Tensor]:

        segments = [d['input_values'] for d in data]
        speech_list = []
        sampling_rate = self.processor.feature_extractor.sampling_rate
        for file, start, end in segments:
            offset = float(start)
            duration = float(end) - offset
            signal, sr = audiofile.read(file, offset=offset, duration=duration)
            speech_list.append(signal)
        input_features = self.processor(
            speech_list, sampling_rate=sampling_rate)
        label_features = [d['labels'] for d in data]
        d_type = torch.long if isinstance(
            label_features[0], int) else torch.float
        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors='pt',
        )
        batch['labels'] = torch.tensor(label_features, dtype=d_type)

        return batch


class CTCTrainer(Trainer):
    r"""
    Trainer for CTC models.

    """

    def __init__(
        self,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        r"""
        Args:
            criterion (:obj:`nn.Module`): The CTC loss function.
            kwargs: The keyword arguments to forward to :obj:`Trainer`. If you're wondering what can be passed here,
                you can find all the hints in Trainer's docstrings.
        Example::
            >>> from datasets import load_metric
            >>> metric = load_metric("wer")
            >>> def compute_metrics(pred):
            ...     pred_logits = pred.predictions
            ...     pred_ids = np.argmax(pred_logits, axis=-1)
            ...     pred.label_ids[pred.label_ids == -100] = num_labels
            ...     wer = metric.compute(predictions=pred_ids, references=pred.label_ids)
            ...     return {"wer": wer}
            >>> model = Wav2Vec2ForCTC.from_pretrained(...)
            >>> criterion = CTCLoss(blank_index=0)
            >>> train_dataset = ...

        """
        super().__init__(**kwargs)

        self.criterion = criterion

    def compute_loss(self, model, inputs, return_outputs=False):
        r"""
        Compute loss against the criterion set on instantiation of the CTCTrainer class.
        """

        labels = inputs.pop('labels') 
        outputs = model(**inputs)
        logits = outputs[0]
        loss = self.criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: typing.Dict[str, typing.Union[torch.Tensor, typing.Any]],
    ) -> torch.Tensor:
        r"""Perform a training step on a batch of inputs."""

        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss = loss.sum()
        # loss.backward()
        self.scaler.scale(loss).backward()

        return loss.detach()


def compute_metrics(p: EvalPrediction):
    r"""
    Compute the metrics for the model for an evaluation loop.
    Metrics are specified by the metrics
    """
    preds = p.predictions[0] if isinstance(
        p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds)

    dim = preds.shape[1]

    scores = {}

    for name, metric in metrics.items():
        score = 0

        for idx in range(dim):
            if idx != 1:
                if idx == 0:
                    print(
                        f'{name} Arousal: {metric(p.label_ids[:, idx], preds[:, idx])}')
                elif idx == 2:
                    print(
                        f'{name} Valence: {metric(p.label_ids[:, idx], preds[:, idx])}')
                score += metric(p.label_ids[:, idx], preds[:, idx])

        scores[name] = score / 2

    return scores


def train_model(train_dataset: Dataset, test_dataset: Dataset = None, dataset_Name: str = 'jl', model_path: str = None):
    r"""
    Trains specified model on specified train and test datasets.
    If model_path is not specfied, will load in audEERING model by default
    """
    # load model from local repo
    file_path = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    root = (os.path.dirname(os.path.dirname(file_path)))
    if (model_path == None):
        model_path = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load wrapped feature extractor and tokenizer from processor object
    processor = Wav2Vec2Processor.from_pretrained(
        os.path.dirname(os.path.realpath(__file__)))
    # Format used by audEERING for their finetuning as of 16-10-2022
    labels = ['LABEL_0', 'LABEL_1', 'LABEL_2']
    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_path,
        config=transformers.AutoConfig.from_pretrained(
            model_path,
            num_labels=len(labels),
            label2id={label: i for i, label in enumerate(labels)},
            id2label={i: label for i, label in enumerate(labels)},
            finetuning_task='emotion',
        )
    ).to(device)  # Move model to GPU if available

    # Create data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Training parameters
    training_args = TrainingArguments(
        output_dir=root + f"/data/{dataset_Name}/training/",
        logging_dir=log_root,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=10,
        save_steps=1,
        eval_steps=1,
        logging_steps=1,
        fp16=True,
        learning_rate=5e-5,
        metric_for_best_model='CCC',
        save_total_limit=3,
        greater_is_better=True,
        load_best_model_at_end=True,
        ignore_data_skip=True,
    )

    # Freeze layers
    model.freeze_feature_extractor()

    # Create trainer with previously defined training arguments
    trainer = CTCTrainer(
        ConcordanceCorCoeff(),
        # Wav2Vec2ForSpeechClassification.from_pretrained(model_name),
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[TensorBoardCallback()],
    )

    # Train model
    trainer.train()
    trainer.save_model(
        f'modelSave-{datetime.datetime.now().strftime("%H-%M_%d-%m-%Y")}')
    return processor, model


def load_model(model_path: str = None):
    r"""
    Loads model from local repo or HuggingFace hub
    """
    if model_path is None:
        model_path = os.path.dirname(os.path.realpath(__file__))
    else:
        file_path = os.path.realpath(os.path.join(
            os.getcwd(), os.path.dirname(__file__)))
        root = (os.path.dirname(file_path))
        model_path = root + model_path
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    print(model_path)

    # model_path = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim' # For HASEL if git lfs is not functional
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_path)
    model.to(device)
    return processor, model

# ----------------- Testing -----------------


class RegressionHead(nn.Module):
    r"""Classification head from audEERING/wav2vec2-how-to as of 16-10-2022."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier from audEERING/wav2vec2-how-to."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2ForSpeechClassification(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
    model: Wav2Vec2ForSpeechClassification = None,
    processor: Wav2Vec2Processor = None,
) -> np.ndarray:
    r"""Predicts emotions or extracts embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y).logits

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y
