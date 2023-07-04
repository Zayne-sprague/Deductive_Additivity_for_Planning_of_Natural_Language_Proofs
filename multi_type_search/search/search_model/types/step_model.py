from step_gen.inference import get_model, get_tokenizer, generate_output
from multi_type_search.search.graph import Graph
from multi_type_search.search.search_model import SearchModel

import torch
from typing import List, Union, Dict, Tuple
import transformers


class StepModel(SearchModel):
    """
    A StepModel is meant to take in multiple statements (usually 2) and generate a new statement from them.  Depending
    on the type of step model parameters loaded, it could be Abductive, Deductive, or something new.
    """
    search_obj_type: str = 'step_model'

    model: torch.nn.Module
    tokenizer: transformers.AutoTokenizer
    device: str

    def __new__(
            cls,
            model_name: str,
            max_output_length: int = 64,
            num_return_sequences: int = 1,
            batch_size: int = 4,
            device: str = 'cpu',
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created, or else returns the previous singleton object"""

        instance_name = f'instance__{model_name}_{max_output_length}_{batch_size}_{device}'
        if force_new_instance:
            return super(StepModel, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(StepModel, cls).__new__(cls))
        return getattr(cls, instance_name)

    def __init__(
            self,
            model_name: str,
            max_output_length: int = 64,
            num_return_sequences: int = 1,
            batch_size: int = 4,
            device: str = 'cpu',
            force_new_instance: bool = False
    ):
        """
        :param model_name: The name of the model (name of the folder in ROOT_PROJECT_DIRECTOR/trained_models)
        :param max_output_length: Integer length of tokens returned by the step model
        :param device: Torch device to run the model on
        :param force_new_instance: This class will try to use a singleton pattern so that you do not load the same model
            twice, you can skip this logic by turning this parameter to true.
        """

        if hasattr(self, 'instantiated'):
            return

        self.model_name = model_name
        self.max_output_length = max_output_length
        self.num_return_sequences = num_return_sequences
        self.batch_size = batch_size
        self.device = device

        self.torch_device = torch.device(device)
        self.model = get_model(model_name, max_length=max_output_length, device=self.torch_device)
        self.tokenizer = get_tokenizer(model_name)
        self.num_return_sequences = num_return_sequences
        self.batch_size = batch_size
        self.instantiated = True

    def sample(
            self,
            text: Union[str, List[str]],
            sample_override=None
    ) -> Union[List[str], List[List[str]]]:
        """
        Generate samples from text.

        :param text: The text you want to generate samples from (usually two sentences)
        :param sample_override: How many samples to generate (this will be used if specified over the class level val)
        :return: A list of samples per text prompt given (if 1 prompt was given only, then only N samples will be
            returned)
        """

        samples = sample_override if sample_override is not None else self.num_return_sequences

        _, generated, _ = generate_output(
            text,
            self.tokenizer,
            self.model,
            self.device,
            batch_size=self.batch_size,
            num_return_sequences=samples
        )

        generated = [output.strip('<pad> ').strip('</s>') for output in generated]

        if isinstance(text, str):
            return generated
        else:
            batches = []
            for gen_idx in range(len(text)):
                batch = []
                for sample_idx in range(samples):
                    batch.append(generated[(gen_idx * samples) + sample_idx])
                batches.append(batch)
            return batches

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
            'model_name': self.model_name,
            'max_output_length': self.max_output_length,
            'num_return_sequences': self.num_return_sequences,
            'device': self.device
        }

    def to(self, device: str) -> 'StepModel':
        if device == self.device:
            return self

        return StepModel(
            self.model_name,
            self.max_output_length,
            self.num_return_sequences,
            self.batch_size,
            device
        )
