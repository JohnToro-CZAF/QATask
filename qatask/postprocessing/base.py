class BasePostProcessor(object):
    """Base class for post processors."""

    def __init__(self, cfg, db_path):
        """Initialize the post processor.

        Args:
            config (dict): Configuration for the post processor.
        """
        self.cfg = cfg
        self.db_path = db_path
    
    def process(self, data):
        """Process the data.

        Args:
            data (dict): Data to process.

        Returns:
            dict: Processed data.
        """
        return data
        
    def __call__(self, data):
        return self.process(data)

