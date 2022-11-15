from transformers import pipeline
from .base import BaseReader
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]
        
class BertReader(BaseReader):
    # class __name__
    def __init__(self, cfg=None, tokenizer=None, db_path=None) -> None:
        super().__init__(cfg, tokenizer, db_path)
        self.cfg = cfg
        self.pipeline = pipeline('question-answering', 
                                 model=self.cfg.model_checkpoint, 
                                 tokenizer=self.cfg.model_checkpoint, 
                                 device="cuda:0", 
                                 batch_size=self.cfg.batch_size)

    def prepare(self, data):
        """
        Args:
          data: A list of {question, contexts:List}
        Returns:
          data: A list of {question, context}
        """
        res = []
        for idx, item in enumerate(data):
            for context in item['contexts']:
                res.append({'question': item['question'], 'context': context})
        return res

    def postprocess(self, prepared, predicted):
        answer = []
        QAans = {'question': "",
                'scores':[], 
                'starts':[], 
                'end':[], 
                'answers':[]}
        #WorkAORUND
        prepared.append({'question': 'dummy', 'context': 'dummy'})

        for idx, QA in enumerate(predicted):
            question = prepared[idx]['question']
            QAans['question'] = question
            QAans['scores'].append(QA['score'])
            QAans['starts'].append(QA['start'])
            QAans['end'].append(QA['end'])
            QAans['answers'].append(QA['answer'])
            if len(predicted) == 1:
                answer.append(QAans)
            else:
                if prepared[idx]['question'] != prepared[idx+1]['question']:
                    answer.append(QAans)
                    QAans = {'question': "",
                      'scores':[], 
                      'starts':[], 
                      'end':[], 
                      'answers':[]}
              
        return answer
    
    def __call__(self, data):
        """
          Args:
            data: A list of question and contexts
            Example: [{"question": "Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", "contexts":["Các nghiên cứu lịch sử cho thấy Cổ Am rất có thể là nơi nữ tướng Lê Chân (? - 43) dựng căn cứ để luyện tập nghĩa quân trong cuộc khởi nghĩa của Hai Bà Trưng (40) chống lại ách đô hộ của nhà Hán.", ]}, {'question':"thành phố biển nào là nơi diễn ra lễ hội rước đèn trung thu lớn nhất việt nam", 'context':'Tết trung thu tại Việt Nam . Tại Phan Thiết ( Bình Thuận) , người ta tổ chức rước đèn quy mô lớn với hàng ngàn học sinh tiểu học và trung học cơ sở diễu hành khắp các đường phố , lễ hội này được xác lập kỷ lục lớn nhất Việt Nam .'}]
          Return: [{'question':"", 'scores':[], 'starts:[]', 'ends:[]', 'answers:[]'}]
            [{'question':"Ai là người đứng đầu trong cuộc chống lại chính quyền ]Đông Hán", 'score': [5.1510567573131993e-05], 'start': [65], 'end': [72], 'answers': ['Lê Chân']}, {'scores': [0.7084577679634094], 'starts': [33], 'ends': [57], 'answers': ['Phan Thiết (Bình Thuận)']}]
        """
        _data = []
        print("Reading candidate passages ...")
        for item in data:
            question = item['question']
            candidate_passages = item['candidate_passages']
            contexts = []
            for doc_id, wikipage in candidate_passages:
                context = self.cur.execute("SELECT text FROM documents WHERE id = ?", (str(doc_id), )).fetchone()        
                contexts.append(context[0])
            _data.append({'question': question, 'contexts': contexts})
        prepared = self.prepare(_data)
        prepared_dataset = ListDataset(prepared)
        predicted = []
        for batch in tqdm(DataLoader(prepared_dataset, batch_size=self.cfg.batch_size, num_workers=4)):
            predicted_batch = self.pipeline(batch)
            predicted.extend(predicted_batch)
        answer = self.postprocess(prepared, predicted)
        saved_format = {'data': []}
        for idx, item in enumerate(answer):
            max_score = max(item['scores'])
            bestans = item['answers'][item['scores'].index(max_score)]
            saved_format['data'].append({'id':'testa_{}'.format(idx+1),
                                          'question':item['question'],
                                          'answer': bestans})
        print("reading done")
        return saved_format
  
if __name__ == "__main__":
    class Config:
        def __init__(self) -> None:
            self.model_checkpoint = "nguyenvulebinh/vi-mrc-large"
            self.batch_size = 8
    
    config = Config()
    reader = BertReader(config, None, "qatask/database/wikipedia_db/wikisqlite.db")
    data = [{'question': 'Ai là đạo diễn phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Ai là đạo diễn phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Ai là đạo diễn phim Titanic', 'candidate_passages': [(1, None)]},{'question': 'Titanic là thuyền gì?', 'candidate_passages': [(1, None)]}, {'question': 'Titanic là thuyền gì?', 'candidate_passages': [(1, None)]}, {'question': 'James Cameron là ai?', 'candidate_passages': [(1, None)]}]
    answer = reader(data)
    print(answer)