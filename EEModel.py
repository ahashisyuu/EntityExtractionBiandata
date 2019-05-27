import tensorflow as tf


def create_eemodel(all_sequence):
    TModel = Baseline2
    model = TModel(all_sequence)
    return model.get_output()


class Model:
    def __init__(self, all_sequence):
        self.all_sequence = all_sequence
        self.output = None
        self.build_model()

    def build_model(self):
        raise NotImplementedError

    def get_output(self):
        return self.output


class Baseline(Model):
    def build_model(self):
        self.output = self.all_sequence[-1]


class Baseline2(Model):
    def build_model(self):
        o1 = self.all_sequence[-1]
        o2 = self.all_sequence[-2]
        o3 = self.all_sequence[-3]
        o4 = self.all_sequence[-4]
        self.output = (o1+o2+o3+o4) / 4


