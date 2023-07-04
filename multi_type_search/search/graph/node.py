from multi_type_search.utils.search_utils import normalize

from typing import List, Dict


class Node:
    """
    A primitive data type in the Graph object which holds the actual string value of the generation/premise/goal as well
    as a variety of other helpful information including it's normalized value, any scores of the generation,
    annotations, and tags.
    """

    value: str
    normalized_value: str
    scores: Dict[str, any]
    annotations: Dict[str, any]
    tags: Dict[str, any] = None
    tmp: Dict[str, any] = None

    def __init__(
            self,
            value: str,
            scores: Dict[str, any] = None,
            annotations: Dict[str, any] = None,
            tags: Dict[str, any] = None,
            tmp: Dict[str, any] = None
    ):
        """
        :param value: The string value of the current node.
        :param scores: Stores the scores of the step used usually for evaluation (not for searching)
        :param annotations: Stores the manually created annotations for a Node
        :param tags: Helpful tags that can be created for a Node (i.e. {step_type: 'deductive'} etc.)
        :param tmp: Temporary storage of information that will never be exported
        """

        self.value = value
        self.normalized_value = normalize(value)
        self.scores = scores if scores else {}
        self.annotations = annotations if annotations else {}
        self.tags = tags if tags else {}
        self.tmp = tmp if tmp else {}

    def to_json(self) -> Dict[str, any]:
        """Helper to convert Nodes into their json structure"""

        out = {'value': self.value}
        if len(self.scores) > 0:
            out["scores"] = self.scores
        if len(self.annotations) > 0:
            out['annotations'] = self.annotations
        if len(self.tags) > 0:
            out['tags'] = self.tags

        return out

    @classmethod
    def from_json(cls, json: Dict[str, any]) -> 'Node':
        """Helper to make a Node instance from a json structure"""

        if 'output' in json:
            return cls.from_canonical_json(json)

        value = json['value']
        scores = json.get("scores", None)
        annotations = json.get("annotations", None)
        tags = json.get("tags", None)
        instance = cls(value, scores, annotations, tags)
        return instance

    def to_canonical_json(self) -> Dict[str, any]:
        """Helper for converting a Node into it's canonical json structure"""
        return {'output': self.value}

    @classmethod
    def from_canonical_json(cls, json: Dict[str, any]) -> 'Node':
        """Helper to make a Node instance from their canonical json structure"""

        value = json['output']
        instance = cls(value)
        return instance

    def __eq__(self, other: any) -> bool:
        return isinstance(other, Node) and other.normalized_value == self.normalized_value

    def __repr__(self) -> str:
        return f'<Node: {self.value}>'
