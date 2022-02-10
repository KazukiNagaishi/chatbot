from transformers import BertJapaneseTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# 埋込み表現を計算
def calc_embedding(text):
    bert_tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_tokens[:126] + ['[SEP]'])
    tokens_tensor = torch.tensor(ids).reshape(1, -1)
    with torch.no_grad():
        output = model_bert(tokens_tensor)

    return output[1].numpy()


# コサイン類似度を計算
def calc_cosine_similarity(text1, text2):
    w1 = calc_embedding(text1)
    w2 = calc_embedding(text2)
    return cosine_similarity(w1, w2)[0, 0]

# Main
def return_reply(message):
    sentences = []
    similarities = []

    query = message

    f = open('Data/data.txt', 'r', encoding='UTF-8')
    while True:
      data = f.readline().strip()
      if data:
          sentences.append(data.split(','))
      else:
          break
    f.close()

    for i, data in enumerate(sentences):
        sim = calc_cosine_similarity(query, data[0])
        similarities.append([sim, i, data[1]])

    similarities.sort(reverse=True)

    return similarities[0][2]

    similarities.clear()
