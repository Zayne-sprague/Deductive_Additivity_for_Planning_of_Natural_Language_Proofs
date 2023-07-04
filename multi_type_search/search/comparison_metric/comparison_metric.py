from multi_type_search.search.search_object import SearchObject

from typing import List, Dict, Tuple


class ComparisonMetric(SearchObject):
    """
    Comparison metrics are used to compare two strings together (can be batched by passing in lists of strings)
    """

    def score(self, group: List[str], targets: List[str], **kwargs) -> List[float]:
        """
        Returns a scalar value for every string to string comparison between the group and target lists.

        ["a", "b", ...] vs ["b", "b", ...]

        for example, should return scalar scores in a list as such

        [COMPARE("a", "b"), COMPARE("b", "b"), COMPARE[..., ...]]

        Where COMPARE is some comparison metric to run against the values.

        :param group: The "left-hand side" of the comparison (what you are comparing)
        :param targets: The "right-hand side" of the comparison (what you are comparing to)
        :return: A list of scalars that represent some comparison between the group and target strings.
        """

        raise NotImplemented('Implement this per evaluation child class.')

    @classmethod
    def config_from_json(cls, type: str, arguments: Dict[str, any], device: str = 'cpu') -> 'ComparisonMetrics':
        from multi_type_search.search.comparison_metric import RougeComparison, ExactComparison, EntailmentComparison, \
            RougeEntailmentHMComparison, SelfRougeComparison, BleuComparison, SelfBleuComparison

        if arguments is None or len(arguments) == 0:
            arguments = {}

        if type == RougeComparison.search_obj_type:
            return RougeComparison(**arguments)
        if type == EntailmentComparison.search_obj_type:
            return EntailmentComparison(**arguments)
        if type == ExactComparison.search_obj_type:
            return ExactComparison()
        if type == RougeEntailmentHMComparison.search_obj_type:
            return RougeEntailmentHMComparison(**arguments)
        if type == SelfRougeComparison.search_obj_type:
            return SelfRougeComparison(**arguments)
        if type == BleuComparison.search_obj_type:
            return BleuComparison()
        if type == SelfBleuComparison.search_obj_type:
            return SelfBleuComparison()

        raise Exception(f"Unknown comparison metrics type in config: {type}")

    def __to_json_config__(self) -> Tuple[str, Dict[str, any]]:
        raise NotImplemented("Please implement the __to_json_config__() func for multiprocessing and ease of saving"
                             " configs.")

    def to_json_config(self) -> Dict[str, any]:
        _type, args = self.__to_json_config__()
        return {
            'constructor_type': 'comparison_metric',
            'type': _type,
            'arguments': args
        }

    def to(self, device: str) -> 'ComparisonMetric':
        return self
