import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from modeling import dropout


def create_eemodel(all_sequence, sent_mask):
    TModel = Baseline2
    model = TModel(all_sequence, sent_mask)
    return model.get_output()


class Model:
    def __init__(self, all_sequence, sent_mask):
        self.all_sequence = all_sequence
        self.sent_mask = sent_mask
        self.output = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def get_output(self):
        return self.output


class Baseline(Model):
    def build_model(self):
        return self.all_sequence[-1]


class Baseline2(Model):
    def build_model(self):
        o1 = self.all_sequence[-1]
        o2 = self.all_sequence[-2]
        o3 = self.all_sequence[-3]
        o4 = self.all_sequence[-4]
        return (o1+o2+o3+o4) / 4


class LSTM(Model):
    def build_model(self):
        temp = self.all_sequence[-1]

        with tf.variable_scope("lstm"):
            seq_len = tf.reduce_sum(self.sent_mask, axis=1)
            gru_fw = GRUCell(num_units=768, activation=tf.tanh)
            gru_bw = GRUCell(num_units=768, activation=tf.tanh)
            outputs, output_states = bidirectional_dynamic_rnn(
                gru_fw, gru_bw, temp,
                sequence_length=seq_len, dtype=tf.float32)

            gru_output = tf.concat(outputs, axis=2)
            gru_output = dropout(gru_output, 0.1)
            gru_output = tf.layers.dense(gru_output, units=768, activation=tf.tanh)

        return temp + gru_output



