import math

from jsonlines import jsonlines
from tqdm import tqdm
import numpy as np

from multi_type_search.search.search_model.types.contrastive import ContrastiveModel
from multi_type_search.utils.paths import DATA_FOLDER, TRAINED_MODELS_FOLDER

index_file = DATA_FOLDER / 'full/hotpot/custom_index/index.npy'
id2index = DATA_FOLDER / 'full/hotpot/custom_index/id2doc.json'

def load_contrastive_model(
        contrastive_model,
        device: str = 'cpu'
) -> ContrastiveModel:
    model = ContrastiveModel.load(TRAINED_MODELS_FOLDER / contrastive_model / 'best_checkpoint.pth', device)
    return model


model = load_contrastive_model('contrastive_hotpot__11_11_12/contrastive_hotpot__2', device='cuda:0')

questions = list(jsonlines.open(str(DATA_FOLDER / 'full/hotpot/meta_qas.json'), 'r'))
corpus = list(jsonlines.open(str(DATA_FOLDER / 'full/hotpot/full_context.json'), 'r'))
# print(corpus[0:2])

#TODO - batch this
# for ex in tqdm(corpus, total=len(corpus), desc='Encoding Corpus'):

encs = []
BATCH_SIZE = 512
for i in tqdm(range(0, len(corpus), BATCH_SIZE), desc='Encoding', total=math.ceil(len(corpus) / BATCH_SIZE)):
    encs.extend(model.get_encodings([x for x in corpus[i:min(len(corpus), i+BATCH_SIZE)]]).detach().tolist())

encs = np.stack(encs)

index_file.parent.mkdir(exist_ok=True, parents=True)
np.save(str(index_file), encs)
