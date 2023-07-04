from multi_type_search.search.graph import Graph, decompose_index
from multi_type_search.search.premise_retriever import PremiseRetriever
from multi_type_search.search.step_selector import StepSelector
from multi_type_search.search.step_type import StepType
from multi_type_search.search.step_validator import ConsanguinityThresholdStepValidator

from copy import deepcopy


def rank_steps(
        graph: Graph,
        step_selector: StepSelector,
        step_type: StepType
):
    step_selector.reset()

    cvalidator = ConsanguinityThresholdStepValidator(threshold=1)

    gold_step = graph.deductions[0].arguments

    _orig_ = deepcopy(graph.premises)
    premises = graph.premises
    deductions = graph.deductions
    abductions = graph.abductions

    graph.deductions = []
    graph.abductions = []
    graph.premises = []

    steps = step_type.generate_step_combinations(graph, premises, [], [])

    graph.premises = premises

    steps = cvalidator.validate(graph, steps)
    steps = [x for x in steps if x.arguments[0] < x.arguments[1]]

    step_selector.add_steps(steps, graph)

    step_selector.iter_size = len(steps)
    ordered_steps = next(step_selector)

    ordered_args = [set(x.arguments) for x in ordered_steps]

    graph.premises = _orig_

    try:
        rank = ordered_args.index(set(gold_step)) + 1
    except Exception:
        try:
            rank = ordered_args.index({gold_step[1], gold_step[0]}) + 1
        except Exception as e:
            print(e)
            print("COULDN'T FIND THE GOLD STEP")
            return []


    graph.deductions = deductions
    graph.abductions = abductions

    return [rank]


def get_rank(graph: Graph, retriever: PremiseRetriever):
    ordering = retriever.reduce(graph.premises, graph.goal, top_n=len(graph.premises), return_scores=True)
    indices = [graph.premises.index(x) for x in ordering]
    gold_indices = [decompose_index(x)[1] for x in graph.deductions[0].arguments]

    ranks = []
    for gold_idx in gold_indices:
        ranks.append(indices.index(gold_idx) + 1)
        indices.remove(gold_idx)
    return ranks


if __name__ == "__main__":
    from multi_type_search.utils.paths import DATA_FOLDER, TRAINED_MODELS_FOLDER
    from multi_type_search.search.premise_retriever import BM25PremiseRetriever, ContrastivePremiseRetriever
    from multi_type_search.search.search_model import NodeEmbedder
    from multi_type_search.search.search_model.types.contrastive import ContrastiveModel
    from multi_type_search.search.step_type import DeductiveStepType
    from multi_type_search.search.step_selector import VectorSpaceSelector, BM25Selector, DPRSelector

    import json
    from tqdm import tqdm

    dataset = DATA_FOLDER / 'full/fever/10_@_10.json'
    data = json.load(dataset.open('r'))
    graphs = [Graph.from_json(x) for x in data]

    # graphs = graphs[0:10]

    # graphs = graphs[3:]
    #for graph in graphs:
    #    print(len(graph.premises))

    def experiment_single_premise_recall():
        bm25_retriever = BM25PremiseRetriever()
        contrastive_model = NodeEmbedder('custom_contrastive_model', device='cuda:0', batch_size=128)
        contrastive_retriever = ContrastivePremiseRetriever(contrastive_model, use_trajectories=False)

        bm25_ranks = []
        contrastive_ranks = []

        for graph in graphs:
            for premise in graph.premises:
                premise.value = premise.value.replace("\t", " ")

        for graph in tqdm(graphs, desc='Running Examples', total=len(graphs)):
            bm25_ranks.extend(get_rank(graph, bm25_retriever))
            contrastive_ranks.extend(get_rank(graph, contrastive_retriever))

        bm25_mrr = sum([1/x for x in bm25_ranks]) / len(bm25_ranks)
        contrastive_mrr = sum([1/x for x in contrastive_ranks]) / len(contrastive_ranks)
        print(f'BM25: {bm25_mrr}')
        print(f'Contrastive: {contrastive_mrr}')

    def experiment_step_recall():
        step_type = DeductiveStepType(None)

        # bm25_step_selector = BM25Selector()
        # dpr_step_selector = DPRSelector(
        #     steps_emb_file=str(DATA_FOLDER / 'full/fever/DPR_10@10_steps.json'),
        #     goals_emb_file=str(DATA_FOLDER / 'full/fever/DPR_10@10_goals.json')
            # steps_emb_file="/mnt/data1/zsprague/tmp/DPR_100@100_steps.json",
            # goals_emb_file="/mnt/data1/zsprague/tmp/DPR_100@100_goals.json"
        # )
        # contrastive_model = NodeEmbedder('contrastive_simcse_finetune_2', device='cuda:0', batch_size=1)
        contrastive_model = NodeEmbedder('contrastive_hotpot_simcse', device='cuda:0', batch_size=1)

        # cm2 = NodeEmbedder('contrastive_simcse_finetune_2', device='cuda:0', batch_size=1, force_new_instance=True)
        # cm2.model.contrastive_model = ContrastiveModel.load(TRAINED_MODELS_FOLDER/'contrastive_simcse_finetune_2/backbone/best_checkpoint.pth', device='cuda:0')
        #
        # cm3 = NodeEmbedder('contrastive_simcse_finetune_2', device='cuda:0', batch_size=1, force_new_instance=True)
        # params = zip(contrastive_model.model.contrastive_model.model.model.state_dict().items(), cm2.model.contrastive_model.model.model.state_dict().items())
        # parameters = {c1_k: (c1_v + c2_v) / 2 for (c1_k, c1_v), (_, c2_v) in params}
        # cm3.model.contrastive_model.model.model.load_state_dict(parameters)
        #
        contrastive_step_selector = VectorSpaceSelector(contrastive_model)
        # cm2_step_selector = VectorSpaceSelector(cm2)
        # cm3_step_selector = VectorSpaceSelector(cm3)

        bm25_ranks = []
        contrastive_ranks = []
        # cm2_ranks = []
        # cm3_ranks = []
        dpr_ranks = []

        for graph in tqdm(graphs, desc='Running Examples', total=len(graphs)):
            # dpr_ranks.extend(rank_steps(graph, dpr_step_selector, step_type))

            for premise in graph.premises:
               premise.value = premise.value.replace("\t", " ")
            # bm25_ranks.extend(rank_steps(graph, bm25_step_selector, step_type))
            contrastive_ranks.extend(rank_steps(graph, contrastive_step_selector, step_type))
            # cm2_ranks.extend(rank_steps(graph, cm2_step_selector, step_type))
            # cm3_ranks.extend(rank_steps(graph, cm3_step_selector, step_type))

        # dpr_mrr = sum([1/x for x in dpr_ranks]) / len(dpr_ranks)
        # bm25_mrr = sum([1/x for x in bm25_ranks]) / len(bm25_ranks)
        contrastive_mrr = sum([1/x for x in contrastive_ranks]) / len(contrastive_ranks)
        # cm2_mrr = sum([1/x for x in cm2_ranks]) / len(cm2_ranks)
        # cm3_mrr = sum([1/x for x in cm3_ranks]) / len(cm3_ranks)
        # print(f'BM25: {bm25_mrr}')
        print(f'No Fine Tuning Contrastive: {contrastive_mrr}')
        # print(f'With Fine Tuning Contrasitve: {cm2_mrr}')
        # print(f'Merged Contrastive: {cm3_mrr}')
        # print(f'DPR: {dpr_mrr}')

    # experiment_single_premise_recall()
    experiment_step_recall()
