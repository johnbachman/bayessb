class ChainLoader(object):
    def __init__(self, chain_filename):
        self.chain_filename = chain_filename

    def load(self):
        raise NotImplementedError()

