import os
from jieba import Tokenizer
from functools import reduce


class Segment(Tokenizer):
    def __init__(self):
        Tokenizer.__init__(self)
        self.stop_words = self.load_stop_word('./jieba/stop.txt')

    def load_stop_word(self, stopwords_path):
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                stop_words = f.readlines()
            stop_words = [line.strip() for line in stop_words]
            self.stop_words = stop_words
            return stop_words
        else:
            Warning(FileNotFoundError(stopwords_path, 'not existed!'))
            return ''

    def seg(self, sentence, return_str=True):
        seg_sentence = sentence.replace(' ', '')
        seg_sentence = self.cut(sentence=sentence)
        if not self.stop_words:
            reg_result = ' '.join(seg_sentence)
        else:
            reg_result = [i for i in seg_sentence if i not in self.stop_words]
            if not reg_result:
                Warning('All words stopped!', 'original:', sentence, 'original segment:', ' '.join(seg_sentence))
            else:
                if return_str:
                    reg_result = reduce(lambda x, y: x + ' ' + y, reg_result)
                else:
                    reg_result = reg_result
        return reg_result


# instance
sgt = Segment()

cut = sgt.seg
load_userdict = sgt.load_userdict
load_stopword = sgt.load_stop_word


if __name__ == '__main__':
    seg = cut('支付宝官方近日发布公告称,从3月26日开始,个人用户在使用支付宝:“信用卡还款”功能时,将对超出免费额度的部分,收取一定比例的服务费。应该说此次收费')
    print(seg)