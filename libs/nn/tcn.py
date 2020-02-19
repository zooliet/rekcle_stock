

class RekcleTCN:
    def __init(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
