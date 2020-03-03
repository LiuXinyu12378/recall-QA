"""
构造召回的模型
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
from cut_sentence import cut
import json

def prepar_recall_datas():
    qa_dict = json.load(open("./corpus/qa_dict.json",encoding="utf-8"))
    q_list = []
    q_cut = []
    for i in qa_dict:
        q_list.append(i)
        q_cut.append(" ".join(qa_dict[i]["cut"])) #分词之后的问题 [sentence,sentence,....]

    tfidf_vec = TfidfVectorizer()
    q_vector = tfidf_vec.fit_transform(q_cut) #得到问题的向量

    #准备搜索的索引
    cp = ci.MultiClusterIndex(q_vector,q_list)

    return tfidf_vec,cp,qa_dict


def get_search_result(input):
    tfidf_vec, cp, qa_dict = prepar_recall_datas()
    entity = []
    input_cut = []
    for word,seg in cut(input,by_word=False,use_seg=True):
        input_cut.append(word)
        if seg == "kc":
            entity.append(word)
    # 1. 得到用户问题的向量
    input_vector = tfidf_vec.transform([" ".join(input_cut)])
    #  2. 计算相似度
    result = cp.search(input_vector,k=2,k_clusters=10,return_distance=True)
    print(result)


if __name__ == '__main__':
    get_search_result("python是什么")


  # "产品经理的课程是只针对IT行业的还是有其他行业相关？": {
  #   "cut": [
  #     "产品经理",
  #     "的",
  #     "课程",
  #     "是",
  #     "只",
  #     "针对",
  #     "it",
  #     "行业",
  #     "的",
  #     "还是",
  #     "有",
  #     "其他",
  #     "行业",
  #     "相关",
  #     "？"
  #   ],
  #   "cut_by_word": [
  #     "产",
  #     "品",
  #     "经",
  #     "理",
  #     "的",
  #     "课",
  #     "程",
  #     "是",
  #     "只",
  #     "针",
  #     "对",
  #     "it",
  #     "行",
  #     "业",
  #     "的",
  #     "还",
  #     "是",
  #     "有",
  #     "其",
  #     "他",
  #     "行",
  #     "业",
  #     "相",
  #     "关",
  #     "？"
  #   ],
  #   "entity": [
  #     "产品经理"
  #   ],
  #   "ans": "技能是相通的，但项目以及业务类型都是互联网行业的，没有传统行业的。互联网行业的待遇要比传统行业高很多"
  # },