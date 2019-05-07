class Xic:
    def __init__(self, data, mz_scales, rt_scales):
        self.data = data
        self.mz_scales = mz_scales
        self.rt_scales = rt_scales

    def to_pickle(self, file: str):
        from .load import to_pickle
        to_pickle(self, file)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __repr__(self):
        return str(self.__class__.__name__) + repr(self.data)

    def copy(self):
        return Xic(self.data.copy(), self.mz_scales.copy(), self.rt_scales.copy())

    @classmethod
    def from_pickle(cls, file: str):
        from .load import from_pickle
        return from_pickle(file)
