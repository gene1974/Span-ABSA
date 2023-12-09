import os

# 读取labeled_data.txt中的数据，以五元组的形式组织，返回数据列表
def add_labeled_triplet_for_transe(file_name = './labeled_data.txt', dump_path = './data/'):
    entity = set()
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            item = line.strip().split('\t')
            target, opinion, polarity, product, star, category, date, labeled = item
            entity |= set([target, opinion, polarity, product, category])
            data.append({
                '产品': product,
                '评价对象': target,
                '评价词': opinion,
                '评价类别': category,
                '评价极性': polarity,
            })
    create_graph(data, list(entity), dump_path)
    return data, entity

# 读取文件中的数据，以五元组的形式组织，返回数据列表
def readDataFile(from_file):
    data_list = []
    polar_dict = {'POS': '正面', 'NEU': '中性', 'NEG': '负面'}
    with open(from_file, 'r') as f:
        for line in f:
            target, opinion, polarity, product, category = line.strip().split('\t')
            data_list.append({
                '产品': product,
                '评价对象': target,
                '评价词': opinion,
                '评价类别': category,
                '评价极性': polar_dict[polarity],
            })
    print('Read data_file: {}, data_list: {}'.format(from_file, len(data_list)))
    return data_list

# 将数据文件五元组中的item全部作为实体提取出来，返回实体列表
def getAllEntity(from_file):
    entity = set()
    polar_dict = {'POS': '正面', 'NEU': '中性', 'NEG': '负面'}
    with open(from_file, 'r') as f:
        for line in f:
            target, opinion, polarity, product, category = line.strip().split('\t')
            polarity = polar_dict[polarity]
            entity |= set([target, opinion, polarity, product, category])
    print('Get entity from data_file: {}, entity: {}'.format(from_file, len(entity)))
    return list(entity)

def addTransEDataFromFile(from_file, to_file):
    os.system('mkdir ' + to_file)
    data_list = readDataFile(from_file)
    entity_list = getAllEntity(from_file)
    # create_graph(data_list, entity_list, to_file)
    dump_entity_cls_type(data_list, entity_list, to_file)
    return data_list, entity_list

def addTransEDataFromFile_small(from_file, to_file, data_list_num = 1000):
    os.system('mkdir ' + to_file)
    data_list = readDataFile(from_file)[:data_list_num]
    entity_list = getAllEntity(from_file)
    # create_graph(data_list, entity_list, to_file)
    dump_entity_cls_type(data_list, entity_list, to_file)
    return data_list, entity_list

def define_entity():
    entities = [
        '产品', '评价对象', '评价词', '评价类别', '评价极性',
    ]
    return entities

def define_relation():
    relations = []
    # Product
    relations.append(['产品', '评价对象', '产品评价对象'])
    relations.append(['产品', '评价词', '产品评价词'])
    # Target
    relations.append(['评价对象', '评价词', '对象评价词'])
    relations.append(['评价对象', '产品', '对象评价产品'])
    relations.append(['评价对象', '评价类别', '对象评价类别'])
    # Opinion
    relations.append(['评价词', '评价对象', '观点评价对象'])
    relations.append(['评价词', '产品', '观点被评价产品'])
    relations.append(['评价词', '评价类别', '观点评价类别'])
    relations.append(['评价词', '评价极性', '观点评价极性'])
    # Category
    relations.append(['评价类别', '评价对象', '类别评价对象'])
    relations.append(['评价类别', '评价词', '类别评价词'])
    # Polarity
    relations.append(['评价极性', '评价词', '正面评价词'])
    relations.append(['评价极性', '评价词', '中性评价词'])
    relations.append(['评价极性', '评价词', '负面评价词'])
    return relations

def create_graph(data, ent_list, dump_path):
    # label (neo4j node label)
    label_list = define_entity()
    label_id_dict = {label_list[i]: i for i in range(len(label_list))}
    
    # relation (egde) type
    relation_type = define_relation()
    rel_list = []
    for i in relation_type:
        if i[2] not in rel_list:
            rel_list.append(i[2])
    rel_id_dict = {rel_list[i]: i for i in range(len(rel_list))}

    # entity (node)
    # ent_list = list(entity)
    ent_id_dict = {ent_list[i]: i for i in range(len(ent_list))}
    
    trip_list = []
    for item in data:
        for rel in relation_type:
            head, tail, edge = rel # ['产品', '评价对象', '评价对象']
            trip_list.append([
                ent_id_dict[item[head]], 
                ent_id_dict[item[tail]], 
                rel_id_dict[edge]
            ])
    print('trip_list:', len(trip_list)) # 4128054
    dump_graph(ent_list, rel_list, trip_list, dump_path)
    return ent_list, rel_list, trip_list

def dump_entity(path, entities):
    with open(path, 'w') as f:
        f.write(str(len(entities)) + '\n')
        for i in range(len(entities)):
            f.write(entities[i] + '\t' + str(i) + '\n')

def dump_trip(path, triples):
    with open(path, 'w') as f:
        f.write(str(len(triples)) + '\n')
        for i in range(len(triples)):
            f.write(str(triples[i][0]) + '\t' + str(triples[i][1]) + '\t' + str(triples[i][2]) + '\n')

import os
import random
def dump_graph(ent_list, rel_list, trip_list, path = './data/'):
    dump_entity(path + 'entity2id.txt', ent_list)
    dump_entity(path + 'relation2id.txt', rel_list)
    random.shuffle(trip_list)
    print('trip_list:', len(trip_list))
    n_train = int(len(trip_list) * 0.8)
    dump_trip(path + 'train2id.txt', trip_list[:n_train])
    dump_trip(path + 'valid2id.txt', trip_list[n_train:])
    dump_trip(path + 'test2id.txt', trip_list[n_train:])
    os.system('cd CommentTransE/data && python n-n.py')

# 对于Bert分类器重新定义实体类型: 产品，价格-正向-评价词，价格-正向-评价对象
def define_entity_cls_type():
    cate_list = ['价格','品质','色泽','口感','包装','分量','物流','售后', '其他']
    polar_list = ['正面', '中性', '负面']
    ent_type_list = ['产品', '评价类别', '评价极性'] + [f'{cate}-{polar}-评价词' for cate in cate_list for polar in polar_list] + [f'{cate}-{polar}-评价对象' for cate in cate_list for polar in polar_list]
    return ent_type_list

def dump_entity_cls_type(data_list, ent_list, to_file):
    ent_type_list = define_entity_cls_type()
    ent_type_dict = {ent_type_list[i]: i for i in range(len(ent_type_list))}
    with open(to_file + 'entity_typeid.txt', 'w') as f:
        for ent_type in ent_type_dict:
            f.write(ent_type + '\t' + str(ent_type_dict[ent_type]) + '\n')

    # entity 2 id
    ent_dict = {ent_list[i]: i for i in range(len(ent_list))}

    # record 
    entity2type_dict = {ent_id: -1 for ent_id in range(len(ent_list))}
    for item in data_list:
        entity2type_dict[ent_dict[item['产品']]] = ent_type_dict['产品']
        entity2type_dict[ent_dict[item['评价类别']]] = ent_type_dict['评价类别']
        entity2type_dict[ent_dict[item['评价极性']]] = ent_type_dict['评价极性']
        entity2type_dict[ent_dict[item['评价词']]] = ent_type_dict[f"{item['评价类别']}-{item['评价极性']}-评价词"]
        entity2type_dict[ent_dict[item['评价对象']]] = ent_type_dict[f"{item['评价类别']}-{item['评价极性']}-评价对象"]
    
    with open(to_file + 'entity2typeid.txt', 'w') as f:
        for ent_id in range(len(ent_list)):
            ent_type = entity2type_dict[ent_id]
            f.write(f'{ent_id}\t{ent_type}\n')
    return 

if __name__ == '__main__':
    # add_labeled_triplet_to_graph('./labeled_data.txt')
    addTransEDataFromFile(from_file = '../knowledgebase/unlabel/deduplicate/data.txt', to_file = './data/')
    addTransEDataFromFile_small(from_file = '../knowledgebase/unlabel/deduplicate/data.txt', to_file = './data_small/')
    # define_entity_cls_type()
    
