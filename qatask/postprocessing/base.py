class BasePostProcessor(object):
    """Base class for post processors."""

    def __init__(self, cfg):
        """Initialize the post processor.

        Args:
            config (dict): Configuration for the post processor.
        """
        self.cfg = cfg

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