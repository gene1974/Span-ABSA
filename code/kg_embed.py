import torch
import torch.nn as nn

from code_wospan.AttentionModel import AttentionPooling

# simpliest version: word embedding avg
class KGEnhancedEmbedLayer(nn.Module):
    def __init__(self, tokenizer, encoder,
                 target_triplet_dict, opinion_triplet_dict, target_ids, opinion_ids, ent_emb_dict, enhanced_target_embed, enhanced_opinion_embed, 
                 ):
        super(KGEnhancedEmbedLayer, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.target_triplet_dict = target_triplet_dict
        self.opinion_triplet_dict = opinion_triplet_dict
        self.target_ids = target_ids
        self.opinion_ids = opinion_ids
        self.ent_emb_dict = ent_emb_dict
        self.enhanced_target_embed = enhanced_target_embed
        self.enhanced_opinion_embed = enhanced_opinion_embed
        self.attn_pooling = AttentionPooling(768, 200)
        # self.linear = nn.Linear(768 * 2, 768)

    def encode_triplets(self):
        self.tokenid_dict = {}
        for entity in self.target_triplet_dict:
            token_ids = self.tokenizer(self.target_triplet_dict[entity], return_tensors='pt', padding=True) # (tail_num, token_size)
            self.tokenid_dict[entity] = token_ids['input_ids']

    def decode_text(self, input_ids):
        text_list = []
        for i in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[i])
            text_list.append(text.replace(' ', '').replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', ''))
        return text_list
    
    def calcu_batch_ent_trip_reps(self, token_ids, ent_spans):
        batch_ent_trip_reps = []
        for i in range(token_ids.shape[0]):
            text = self.tokenizer.decode(token_ids[i])
            text = text.replace(' ', '').replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
            ent_trip_reps = []
            for ent_span in ent_spans[i]:
                ent_text = text[ent_span[0]: ent_span[1]]
                # ent_tail_state = self.enhanceByTriplet(ent_text) # (1, hidden_size)
                ent_tail_state = self.enhanceByTransE(ent_text) # (1, hidden_size)
                ent_trip_reps.append(ent_tail_state)
            ent_trip_reps = torch.cat(ent_trip_reps, dim = 0) # (max_num_ents, hidden_size)
            batch_ent_trip_reps.append(ent_trip_reps)
        batch_ent_trip_reps = torch.stack(batch_ent_trip_reps) # (batch_size, max_num_ents, hidden_size)
        return batch_ent_trip_reps
    
    def calcu_batch_ent_trip_repsV2(self, token_ids, ent_spans):
        batch_ent_trip_reps = []
        for i in range(token_ids.shape[0]):
            text = self.tokenizer.decode(token_ids[i])
            text = text.replace(' ', '').replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
            ent_trip_reps = []
            for ent_span in ent_spans[i]:
                ent_text = text[ent_span[0]: ent_span[1]]
                ent_tail_state = self.enhanceByTriplet(ent_text) # (1, hidden_size)
                ent_trip_reps.append(ent_tail_state)
            ent_trip_reps = torch.cat(ent_trip_reps, dim = 0) # (max_num_ents, hidden_size)
            batch_ent_trip_reps.append(ent_trip_reps)
        batch_ent_trip_reps = torch.stack(batch_ent_trip_reps) # (batch_size, max_num_ents, hidden_size)
        return batch_ent_trip_reps
    
    # def enhanceByTopTriplet(self, entity):

    def enhanceByTransE(self, entity):
        if entity in self.ent_emb_dict:
            return self.ent_emb_dict[entity].unsqueeze(0).cuda()
        else:
            return torch.zeros(1, 768).cuda()
    
    def enhanceByTriplet(self, entity):
        enhanced_emb = torch.zeros(1, 768)
        if entity in self.target_ids:
            enhanced_emb += self.enhanced_target_embed[self.target_ids[entity]]
        if entity in self.opinion_ids:
            enhanced_emb += self.enhanced_opinion_embed[self.opinion_ids[entity]]
        return enhanced_emb.cuda()
    
    # def enhanceByTriplet(self, entity):
    #     enhanced_emb = torch.zeros(1, 768 * 2)
    #     if entity in target_ids:
    #         enhanced_emb[:768] += self.enhanced_target_embed[target_ids[entity]]
    #     if entity in opinion_ids:
    #         enhanced_emb[768:] += self.enhanced_opinion_embed[opinion_ids[entity]]
    #     return self.linear(enhanced_emb.cuda())
    
    # 使用三元组信息进行加权
    def enhanceByTripletV2(self, entity):
        tail_list = [entity]
        if entity in self.target_triplet_dict:
            tail_list += self.target_triplet_dict[entity]
        if entity in self.opinion_triplet_dict:
            tail_list += self.opinion_triplet_dict[entity]
        tail_embeds = torch.zeros(1, 768)
        for i in range(len(tail_list)):
            tail_ids = self.tokenizer(tail_list[i], return_tensors='pt', padding=True).to('cuda')
            tail_embed = self.encoder(tail_ids['input_ids'], tail_ids['attention_mask'])[0]
            tail_embed = tail_embed.detach().cpu()
            tail_embeds += torch.mean(tail_embed, dim=0)
        # tail_embeds = torch.stack(tail_embeds)
        # tail_ids = self.tokenizer(tail_list, return_tensors='pt', padding=True).to('cuda') # (batch_size, max_text_len)
        # tail_embeds = self.encoder(tail_ids['input_ids'], tail_ids['attention_mask'])[0]
        # tail_embeds = self.attn_pooling(tail_embeds)
        return tail_embeds
    
    def forward(self, entity):
        return self.enhanceByTransE(entity)


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

# get all entity and its related triplet: entity -> [tail]
def getTriplet(from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'):
    triplet_dict = {}
    with open(from_file, 'r') as f:
        for line in f:
            target, opinion, polarity, product, category = line.strip().split('\t')
            triplet_dict['{}-{}'.format(target, opinion)] = '{}-{}-{}-{}'.format(target, opinion, category, polarity)
    print('Get triplet from data_file: {}, triplet_dict: {}'.format(from_file, len(triplet_dict))) # triplet_dict: 148600
    return triplet_dict

# get all entity and its related triplet: entity -> [tail]
def getTopEntityTriplet(from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'):
    target_dict = {}
    opinion_dict = {}
    with open(from_file, 'r') as f:
        for line in f:
            target, opinion, polarity, product, category = line.strip().split('\t')
            target_dict[target] = target_dict.get(target, {})
            target_dict[target][opinion] = target_dict[target].get(opinion, 0) + 1
            opinion_dict[opinion] = opinion_dict.get(opinion, {})
            opinion_dict[opinion][target] = opinion_dict[opinion].get(target, 0) + 1
    for target in target_dict:
        target_dict[target] = sorted(target_dict[target].items(), key = lambda x: x[1], reverse = True)[:10]
        target_dict[target] = [t[0] for t in target_dict[target]]
    for opinion in opinion_dict:
        opinion_dict[opinion] = sorted(opinion_dict[opinion].items(), key = lambda x: x[1], reverse = True)[:10]
        opinion_dict[opinion] = [t[0] for t in opinion_dict[opinion]]
    # target_dict['花生']: [('不错', 24), ('很好', 18), ('可以', 9), ('一般', 8), ('很不错', 8), ('还可以', 8), ('新鲜', 7), ('不新鲜', 6), ('不好', 6), ('好吃', 6)]
    print('Get entity top triplets, target_dict: {}, opinion_dict: {}'.format(from_file, len(target_dict), len(opinion_dict))) # target_dict: 12122, opinion_dict: 34028
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

# load pre-saved embedes
def loadBertEmbed():
    import pickle
    enhanced_target_embed, enhanced_opinion_embed, target_list, opinion_list = pickle.load(open('/home/gene/Documents/Sentiment/JDComment_seg/data/enhanced_entity_embed_bert.pkl', 'rb'))
    return enhanced_target_embed, enhanced_opinion_embed, target_list, opinion_list

# for kg enhanced embedding
def load_kg_info(model_time = '12111305'):
    from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'
    target_ids, opinion_ids, target_list, opinion_list = getEntity(from_file)
    target_triplet_dict, opinion_triplet_dict = getAllEntityTriplet()
    enhanced_target_embed, enhanced_opinion_embed, target_list, opinion_list = loadBertEmbed()
    target_top_triplet_dict, opinion_top_triplet_dict = getTopEntityTriplet()
    
    # use transe embedding
    import pickle
    ent_dict, ent_emb = pickle.load(open('/home/gene/Documents/Senti/Comment/CommentTransE/result/transe_ent_emb.{}.pkl'.format(model_time), 'rb'))
    ent_emb_dict = {ent: ent_emb[ent_dict[ent]] for ent in ent_dict}

    return target_triplet_dict, opinion_triplet_dict, target_ids, opinion_ids, ent_emb_dict, enhanced_target_embed, enhanced_opinion_embed


if __name__ == '__main__':
    # getBertEmbed()
    # from_file = '/home/gene/Documents/Senti/Comment/knowledgebase/unlabel/deduplicate/data.txt'
    # target_triplet_dict, opinion_triplet_dict = getAllEntityTriplet(from_file)
    # print(max([len(target_triplet_dict[t]) for t in target_triplet_dict]), max([len(opinion_triplet_dict[t]) for t in opinion_triplet_dict]))
    pass

