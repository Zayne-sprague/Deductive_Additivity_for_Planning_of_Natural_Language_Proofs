from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER
from multi_type_search.search.search_model import SearchModel

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import torch
from datasets import Dataset
from typing import Union, List, Dict, Tuple
from enum import Enum


class EntailmentModel(SearchModel):
    """
    Entailment Models are used for determining if a sentence entails, contradicts, or is neutral to another
    sentence.
    """

    search_obj_type: str = 'entailment_model'

    def __new__(
            cls,
            model_name: str,
            device: str = 'cpu',
            batch_size: int = 4,
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created, or else returns the previous singleton object"""

        instance_name = f'instance__{model_name}_{batch_size}_{device}'
        if force_new_instance:
            return super(EntailmentModel, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(EntailmentModel, cls).__new__(cls))
        return getattr(cls, instance_name)

    def __init__(
            self,
            model_name: str,
            device: str = 'cpu',
            batch_size: int = 4,
            force_new_instance: bool = False
    ):
        """
        :param model_name: Name of the model stored in {ROOT_OF_DIRECTORY}/trained_models
        :param device: Name of the torch device to load the model onto.
        :param batch_size: Number of batches to use when training/running inference
        :param force_new_instance: This class will try to use a singleton pattern so that you do not load the same model
            twice, you can skip this logic by turning this parameter to true.
        """
        if hasattr(self, 'instantiated'):
            return

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = AutoModelForSequenceClassification.from_pretrained(TRAINED_MODELS_FOLDER / model_name)

        if 'cuda' in device and torch.cuda.is_available():
            self.model.to(device)
        else:
            self.model.to('cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODELS_FOLDER / model_name)

        self.batch_size = batch_size

        self.instantiated = True

    def score(self, targets: Union[List[str], str], predictions: Union[List[str], str]):
        """

        :param targets: List of target sentences you want to see if is entailed by a prediction
        :param predictions: A list of predictions you want to compare against the target
        :return: The ENTAILMENT probability per prediction target combo
        """

        if isinstance(predictions, str):
            predictions = [predictions] * (1 if isinstance(targets, str) else len(targets))
        if isinstance(targets, str):
            targets = [target] * len(predictions)

        dataset = Dataset.from_dict({"inputs": predictions, "targets": targets})
        dataset = dataset.map(lambda e: e, batched=True)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        all_probabilities = []
        for batch in dataloader:
            inputs_encoded = self.tokenizer(
                batch['inputs'],
                text_pair=batch['targets'],
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            inputs_encoded.data = {
                k: v.to(self.model.device)
                if isinstance(v, torch.Tensor) else
                v
                for k, v in inputs_encoded.data.items()
            }

            # This model will produce 3 logits for 3 classes ENTAIL, NEUTRAL, and CONTRADICT
            # We are only interested in the entailment class probability usually which is why we only look at the last
            # index
            logits = self.model(**inputs_encoded).logits.detach().cpu()
            probs = torch.nn.functional.softmax(logits, dim=1)[:, -1].tolist()
            all_probabilities.extend(probs)

        return all_probabilities

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
                'model_name': self.model_name,
                'device': self.device,
                'batch_size': self.batch_size
            }

    def to(self, device: str) -> 'EntailmentModel':
        if self.device == device:
            return self

        return EntailmentModel(self.model_name, device, batch_size=self.batch_size)
