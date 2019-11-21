import tensorflow_hub as hub
import numpy as np
import tensorflow_text


def test_use_multi_2():
    # Some texts of different lengths.
    english_sentences = ["dog", "Puppies are nice.",
                         "I enjoy taking long walks along the beach with my dog."]
    italian_sentences = ["cane", "I cuccioli sono carini.",
                         "Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane."]
    japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]

    embed = hub.load("./universal-sentence-encoder-multilingual_2")

    # Compute embeddings.
    en_result = embed(english_sentences)["outputs"]
    it_result = embed(italian_sentences)["outputs"]
    ja_result = embed(japanese_sentences)["outputs"]

    # Compute similarity matrix. Higher score indicates greater similarity.
    similarity_matrix_it = np.inner(en_result, it_result)
    similarity_matrix_ja = np.inner(en_result, ja_result)


def test():
    input_sentences = "兰州西车辆段2019年度非标配件委外制作项目招标二次公告,且末县2019年国有贫困林场扶贫资金项目".split(
        ',')

    history_sentences = "兰州西车辆段2019年度非标配件委外制作项目招标二次公告,兰州西车辆段2019年度非标配件委外制作项目单一来源谈判结果公示,且末县2019年国有贫困林场扶贫资金项目,且末县2019年国有贫困林场扶贫资金项目中标候选人公示".split(
        ',')

    embed = hub.load("./universal-sentence-encoder-multilingual_2")

    input_result = embed(input_sentences)["outputs"].numpy().tolist()
    history_result = embed(history_sentences)["outputs"].numpy().tolist()

    zh_similarity_matrix = np.inner(input_result, history_result)

    print(zh_similarity_matrix)

'''
bert向量批量计算最相似
0:52:13
我的手机 2019-11-14 0:52:13
规则加向量计算共同约数

'''

if __name__ == "__main__":
    test()
