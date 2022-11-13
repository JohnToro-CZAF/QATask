class BasePostProcessor(object):
    """Base class for post processors."""

    def __init__(self, cfg):
        """Initialize the post processor.

        Args:
            config (dict): Configuration for the post processor.
        """
        self.cfg = cfg

    def checktype(self, text):
        isdate = False
        return isdate
    
    def date_transform(self, text):
        return text

    def postprocess(self, text):
        if text == '':
            return 'null'
        anstype = self.checktype(text)
        if anstype == 1:
            return self.date_transform(text)
        elif anstype == 2:
            # Only number
            return text
        elif anstype == 3:
            return 'wiki/'+text.replace(" ", "_")
        else:
            assert("Error")

    def process(self, data):
        """Process the data.

        Args:
            data (dict): List of {question, answers:List, scores:List}

        Returns:
            dict: Processed data.
        """
        saved_format = {'data': []}
        for item, idx in enumerate(data):
            bestans = item['answer'][item['scores'].argmax()]
            bestans = self.postprocess(bestans)
            saved_format['data'].append({'id':'testa_{}'.format(idx+1),
                                         'question':item['question'],
                                         'answer': bestans})
        return saved_format

    def __call__(self, data):
        return self.process(data)