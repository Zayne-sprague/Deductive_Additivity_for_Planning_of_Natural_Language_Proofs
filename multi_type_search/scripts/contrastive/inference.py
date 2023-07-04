from multi_type_search.search.search_model.types.contrastive import ContrastiveModel, NonParametricVectorSpace
from multi_type_search.search.search_model.types.contrastive.contrastive_utils import cosine_similarity_metric
from multi_type_search.scripts.contrastive.evaluate import iterative_search, iterative_search__subt
from multi_type_search.utils.paths import TRAINED_MODELS_FOLDER


def load_contrastive_model(
        contrastive_model,
        device: str = 'cpu'
) -> ContrastiveModel:
    model = ContrastiveModel.load(TRAINED_MODELS_FOLDER / contrastive_model / 'best_checkpoint.pth', device)
    return model


def infer(
        model: NonParametricVectorSpace,
        args,
        goal,
        arg_tags = (),
        top_k: int = 10,
        use_subt: bool = False,
):
    arg_toks = model.tokenize(args)
    goal_tok = model.tokenize(goal)

    arg_emb = model(arg_toks)[0]
    goal_emb = model(goal_tok)[0]

    if use_subt:
        candidates = iterative_search__subt(arg_emb, goal_emb, top_k=top_k)
    else:
        candidates = iterative_search(arg_emb, goal_emb, top_k=top_k)

    for idx, c in enumerate(reversed(candidates)):
        print(f"====== {len(candidates) - idx} : {c[1]:.6f} =======")
        print(f'\t{arg_tags[c[0][0]] + ":: " if len(arg_tags) > 0 else ""}{args[c[0][0]]}')
        print(f'\t{arg_tags[c[0][1]] + ":: " if len(arg_tags) > 0 else ""}{args[c[0][1]]}')
        print('====================')

