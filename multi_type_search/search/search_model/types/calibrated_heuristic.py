from multi_type_search.search.search_model import SearchModel
from multi_type_search.search.graph import Graph
from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER

from typing import List, Dict, Tuple
import torch
import transformers
from datasets import disable_progress_bar
disable_progress_bar()


def chunks(it, n):
    curr = []
    for x in it:
        curr.append(x)
        if len(curr) == n:
            yield curr
            curr = []
    if len(curr) > 0:
        yield curr


class CalibratorHeuristic(SearchModel):
    """
    Given statements, predict a scalar value (score) for how "good" those statements are towards generating a useful
    statement in a stepmodel.  Depending on the way the model was trainined, it could evaluate the potential score of
    abductive or deductive steps.
    """

    search_obj_type: str = 'calibrator_heuristic'

    def __init__(
            self,
            model_name: str,
            device: str = 'cpu',
            goal_conditioned: bool = False,
            batch_size: int = 4,
            force_new_instance: bool = False
    ):
        """
        :param model_name: Name of the model stored in {ROOT_OF_DIRECTORY}/trained_models
        :param device: Name of the torch device to load the model onto.
        :param goal_conditioned: Should the model should include the goal as part of the scoring prompt.
        :param batch_size: Number of batches to use when training/running inference
        :param force_new_instance: This class will try to use a singleton pattern so that you do not load the same model
            twice, you can skip this logic by turning this parameter to true.
        """

        if hasattr(self, 'instantiated'):
            return

        self.model_name = model_name
        self.device = device
        self.goal_conditioned = goal_conditioned
        self.batch_size = batch_size

        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(TRAINED_MODELS_FOLDER / model_name)
        self.model.eval()

        if 'cuda' in device and not torch.cuda.is_available():
            device = 'cpu'

        self.model.to(device)

        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(TRAINED_MODELS_FOLDER / model_name)

        self.instantiated = True

    def __new__(
            cls,
            model_name: str,
            device: str = 'cpu',
            goal_conditioned: bool = False,
            batch_size: int = 4,
            force_new_instance: bool = False
    ):
        """ creates a singleton object, if it is not created, or else returns the previous singleton object"""

        instance_name = f'instance__{model_name}_{goal_conditioned}'
        if force_new_instance:
            return super(CalibratorHeuristic, cls).__new__(cls)

        if not hasattr(cls, instance_name):
            setattr(cls, instance_name, super(CalibratorHeuristic, cls).__new__(cls))
        return getattr(cls, instance_name)

    def score_steps(
            self,
            graph: Graph,
            steps: List['Step'],
            *args,
            **kwargs
    ) -> List[float]:
        step_inputs = [" ".join([graph[x].normalized_value for x in step.arguments]) for step in steps]
        batches = [
            self.tokenizer(
                chunk_inputs,
                text_pair=[graph.goal.normalized_value]*len(chunk_inputs) if self.goal_conditioned else None,
                truncation=True, padding=True, return_tensors='pt'
            ) for chunk_inputs in chunks(step_inputs, self.batch_size)
        ]

        scores = []

        for batch in batches:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            logits = self.model(**batch, return_dict=True).logits
            for batch_step in range(logits.shape[0]):
                scores.append(logits[batch_step, 1].item())
            del logits
        return scores

    @classmethod
    def from_config(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'CalibratorHeuristic':

        if arguments is None or len(arguments) == 0:
            arguments = {}

        if type == cls.search_obj_type:
            return CalibratorHeuristic(**arguments, device=device)

        raise Exception(f"Unknown calibrator heuristic type in config: {type}")

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        return self.search_obj_type, {
                'model_name': self.model_name,
                'device': self.device,
                'goal_conditioned': self.goal_conditioned,
                'batch_size': self.batch_size
            }

    def to(self, device: str) -> 'CalibratorHeuristic':
        if device == self.device:
            return self

        return CalibratorHeuristic(
            self.model_name,
            device,
            self.goal_conditioned,
            self.batch_size
        )
