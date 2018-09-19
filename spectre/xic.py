class Xic:
    def __init__(self, data, rts, mz_scales):
        self.data = data
        self.rts = rts
        self.mz_scales = mz_scales

    def to_pickle(self, file: str):
        from spectre.load import to_pickle
        to_pickle(self, file)
