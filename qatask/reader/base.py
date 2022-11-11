class BaseReader:
    def __init__(self, cfg, tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
    
    def __call__(self, data):
        for question in data:
            #question['candidate_passages'] -> [(doc_id 1, wikipage 1), ..., (doc_id topk, wikipage topk)]
            question['answer'] = question['candidate_passages'][0][1]
            question.pop('candidate_passages', None)
        return data