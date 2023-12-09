import torch

# simpliest version: word embedding avg

# get all entity: entity -> id
def getEntity(from_file):
    target_list = []
    opinion_list = []
    target_ids = {} # entity -> id
    opinion_ids = {}
    with open(from_file, 'r') as f:
        for line in f:
            target, opinion, polarity, product, category = line.strip().split('\t')
            if target not in target_ids:
                target_ids[target] = len(target_ids)
                target_list.append(target)
            if opinion not in opinion_ids:
                opinion_ids[opinion] = len(opinion_ids)
                opinion_list.append(opinion)
    print('Get entity from data_file: {}, target_ids: {}, opinion_ids: {}'.format(from_file, len(target_ids), len(opinion_ids))) # target_dict: 12122, opinion_dict: 34028
    return target_ids, opinion_ids, target_list, opinion_list

# get all entity and its related triplet: entity -> [tail]
def getAllEntityTriplet(from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'):
    target_dict = {}
    opinion_dict = {}
    with open(from_file, 'r') as f:
        for line in f:
            target, opinion, polarity, product, category = line.strip().split('\t')
            target_dict[target] = target_dict.get(target, set())
            target_dict[target].add(opinion)
            opinion_dict[opinion] = opinion_dict.get(opinion, set())
            opinion_dict[opinion].add(target)
    print('Get entity triplet from data_file: {}, target_dict: {}, opinion_dict: {}'.format(from_file, len(target_dict), len(opinion_dict))) # target_dict: 12122, opinion_dict: 34028
    return target_dict, opinion_dict

# get single entity embedding
def embedEntity(tokenizer, model, entity):
    token_ids = tokenizer.encode(entity, return_tensors = 'pt', padding = True)
    embeds = model(token_ids.cuda())[0]
    embeds = torch.mean(embeds, dim = 1)
    embeds = embeds.detach().cpu()
    return embeds

def embedAllEntity(entity_list, tokenizer, model):
    entity_embeds = torch.zeros(len(entity_list), model.config.hidden_size)
    for i in range(len(entity_list)):
        entity_embeds[i] = embedEntity(tokenizer, model, entity_list[i])
    # for i in range(0, (len(entity_list) + 16) // 16):
    #     entity_embeds[i * 16: (i + 1) * 16] = embedEntity(tokenizer, model, entity_list[i * 16: (i + 1) * 16])
    print('Embed all entity: {}'.format(entity_embeds.size()))
    return entity_embeds

# emhanced_emb: [entity_emb; avg(tail_emb)]
def enhanceTripEmbed(entity_ids, tail_ids, entity_embeds, tail_embeds, entity_triplet_dict, tail_triplet_dict):
    emb_size = entity_embeds.size(-1)
    enhanced_entity_embeds = torch.zeros(len(entity_ids), 2 * emb_size)
    for entity in entity_ids:
        enhanced_entity_embeds[entity_ids[entity], :emb_size] = entity_embeds[entity_ids[entity]]
        for tail in entity_triplet_dict[entity]:
            enhanced_entity_embeds[entity_ids[entity], emb_size:] += tail_embeds[tail_ids[tail]]
    return enhanced_entity_embeds

def getBertEmbed(from_file):
    from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'
    target_ids, opinion_ids, target_list, opinion_list = getEntity(from_file)
    target_triplet_dict, opinion_triplet_dict = getAllEntityTriplet(from_file) # entity -> [tail]

    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('/data/pretrained/bert-base-chinese/')
    model = BertModel.from_pretrained('/data/pretrained/bert-base-chinese/').cuda()

    target_embed = embedAllEntity(target_list, tokenizer, model)
    opinion_embed = embedAllEntity(opinion_list, tokenizer, model)
    import pickle
    with open('../data/target_embed_bert.pkl', 'wb') as f:
        pickle.dump(target_embed, f)
    with open('../data/opinion_embed_bert.pkl', 'wb') as f:
        pickle.dump(opinion_embed, f)

    # import pickle
    # target_embed = pickle.load(open('../data/target_embed_bert.pkl', 'rb'))
    # opinion_embed = pickle.load(open('../data/opinion_embed_bert.pkl', 'rb'))

    enhanced_target_embed = enhanceTripEmbed(target_ids, opinion_ids, target_embed, opinion_embed, target_triplet_dict, opinion_triplet_dict)
    enhanced_opinion_embed = enhanceTripEmbed(opinion_ids, target_ids, opinion_embed, target_embed, opinion_triplet_dict, target_triplet_dict)
    import pickle
    with open('../data/enhanced_entity_embed_bert.pkl', 'wb') as f:
        pickle.dump([enhanced_target_embed, enhanced_opinion_embed, target_list, opinion_list], f)

def loadBertEmbed():
    import pickle
    enhanced_target_embed, enhanced_opinion_embed, target_list, opinion_list = pickle.load(open('/home/gene/Documents/Sentiment/JDComment_seg/data/enhanced_entity_embed_bert.pkl', 'rb'))
    return enhanced_target_embed, enhanced_opinion_embed, target_list, opinion_list

if __name__ == '__main__':
    # getBertEmbed()
    from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'
    target_triplet_dict, opinion_triplet_dict = getAllEntityTriplet(from_file)
    print(max([len(target_triplet_dict[t]) for t in target_triplet_dict]), max([len(opinion_triplet_dict[t]) for t in opinion_triplet_dict]))

