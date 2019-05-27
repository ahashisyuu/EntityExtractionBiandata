# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT_88.5 on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import tensorflow_estimator as tfes
import pandas as pd

from tensorflow.contrib import tpu
from EvalHook import EvalHook, write_prediction

flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters

bert_dir = "./chinese_L-12_H-768_A-12/"
output_dir = "model_output"
max_seq_length = 300
max_query_length = 10
max_answer_length = 20
learning_rate = 5e-5
do_train = True
do_predict = False
train_batch_size = 8
predict_batch_size = 8
num_train_epochs = 10
checkpoints_steps = 500

flags.DEFINE_string(
    "bert_config_file", bert_dir + "bert_config.json",
    "The config json file corresponding to the pre-trained BERT_88.5 model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", bert_dir + "vocab.txt",
                    "The vocabulary file that the BERT_88.5 model was trained on.")

flags.DEFINE_string(
    "output_dir", output_dir,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", None,
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", bert_dir + "bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT_88.5 model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", max_seq_length,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", max_query_length,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("do_train", do_train, "Whether to run training.")

flags.DEFINE_bool("do_predict", do_predict, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", train_batch_size, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", predict_batch_size,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", learning_rate, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", num_train_epochs,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", checkpoints_steps,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", checkpoints_steps,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", max_answer_length,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class SquadExample(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 sent,
                 question_text,
                 orig_answer_text=None):
        self.qas_id = qas_id
        self.sent = sent
        self.question_text = question_text
        self.orig_answer_text = orig_answer_text


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 answer_text,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.answer_text = answer_text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def compute_position(sent, entity):
    for i in range(len(sent)-len(entity)+1):
        if sent[i] == entity[0]:
            flag = True
            for j in range(1, len(entity)):
                if sent[i+j] != entity[j]:
                    flag = False
                    break
            if flag is True:
                return i
    return -1


# hoka_test = []


def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""

    train_data = pd.read_csv(input_file,
                             header=None,
                             names=["id", "sent", "entity", "label"])
    examples = []
    for item in train_data.iterrows():
        index = item[0]
        data_row = item[1]

        question_text = data_row["entity"]
        orig_answer_text = None
        if is_training:
            orig_answer_text = data_row["label"]
        # else:
        #     if data_row["entity"] == "其他":
        #         print(data_row["id"])
        #         # hoka_test.append(data_row["id"])
        #         continue

        example = SquadExample(
            qas_id=data_row["id"],
            sent=data_row["sent"],
            question_text=question_text,
            orig_answer_text=orig_answer_text)
        examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    for (example_index, example) in enumerate(examples):

        try:

            doc_tokens = tokenizer.tokenize(example.sent)
            query_tokens = tokenizer.tokenize(example.question_text)
        except ValueError:
            print(example.qas_id)
            continue

        if len(doc_tokens) > max_seq_length-len(query_tokens):
            doc_tokens = doc_tokens[:max_seq_length-len(query_tokens)]
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        # print(doc_tokens)
        # print(query_tokens)
        # print(answer_tokens)
        if is_training:
            answer_tokens = tokenizer.tokenize(example.orig_answer_text)
            pos = compute_position(doc_tokens, answer_tokens)
            if pos == -1:
                print(doc_tokens)
                print(answer_tokens)
                print(f"Can't find answer {example.orig_answer_text} in {example.sent}. Because of truncation")
                continue
            start_position = pos
            end_position = pos + len(answer_tokens) - 1
        else:
            start_position = -1
            end_position = -1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if is_training:
            start_position += len(tokens)
            end_position += len(tokens)

        for token in doc_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length + 3:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length + 3
        assert len(input_mask) == max_seq_length + 3
        assert len(segment_ids) == max_seq_length + 3

        if example_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.qas_id))
            tf.logging.info("example_index: %s" % (example_index))
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if is_training:
                answer_text = " ".join(tokens[start_position:(end_position + 1)])
                tf.logging.info("start_position: %d" % (start_position))
                tf.logging.info("end_position: %d" % (end_position))
                tf.logging.info(
                    "answer: %s" % (tokenization.printable_text(answer_text)))

        feature = InputFeatures(
            unique_id=example.qas_id,
            example_index=example_index,
            tokens=tokens,
            answer_text=example.orig_answer_text,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position)

        # Run callback
        output_fn(feature)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]

    output_weights = tf.get_variable(
        "cls/squad/output_weights", [2, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

    final_hidden_matrix = tf.reshape(final_hidden,
                                     [batch_size * seq_length, hidden_size])
    logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    logits = tf.reshape(logits, [batch_size, seq_length, 2])
    logits = tf.transpose(logits, [2, 0, 1])

    unstacked_logits = tf.unstack(logits, axis=0)

    (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

    return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tfes.estimator.ModeKeys.TRAIN)

        (start_logits, end_logits) = create_model(
            bert_config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
        #                     init_string)

        output_spec = None
        if mode == tfes.estimator.ModeKeys.TRAIN:
            seq_length = modeling.get_shape_list(input_ids)[1]

            def compute_loss(logits, positions):
                one_hot_positions = tf.one_hot(
                    positions, depth=seq_length, dtype=tf.float32)
                log_probs = tf.nn.log_softmax(logits, axis=-1)
                loss = -tf.reduce_mean(
                    tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
                return loss

            start_positions = features["start_positions"]
            end_positions = features["end_positions"]

            start_loss = compute_loss(start_logits, start_positions)
            end_loss = compute_loss(end_logits, end_positions)

            total_loss = (start_loss + end_loss) / 2.0

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tfes.estimator.ModeKeys.PREDICT:

            # outer = tf.matmul(tf.expand_dims(tf.nn.softmax(start_logits), axis=2),
            #                   tf.expand_dims(tf.nn.softmax(end_logits), axis=1))
            # outer = tf.matrix_band_part(outer, -1, 15)  # 取上3角15条对角线，表示答案最大长度只能取到15+1个单词
            # yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)  # 寻找最大值在L1轴的索引
            # yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

            predictions = {
                "unique_ids": unique_ids,
                "start_logits": start_logits,
                "end_logits": end_logits,
                # "yp1": yp1,
                # "yp2": yp2,
            }
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "unique_ids": tf.FixedLenFeature([], tf.int64),
        "input_ids": tf.FixedLenFeature([seq_length + 3], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length + 3], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length + 3], tf.int64),
    }

    if is_training:
        name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
        name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if FLAGS.verbose_logging:
            tf.logging.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if FLAGS.verbose_logging:
            tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if FLAGS.verbose_logging:
            tf.logging.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


class FeatureWriter(object):
    """Writes InputFeature to TF example file."""

    def __init__(self, filename, is_training):
        self.filename = filename
        self.is_training = is_training
        self.num_features = 0
        self._writer = tf.python_io.TFRecordWriter(filename)

    def process_feature(self, feature):
        """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
        self.num_features += 1

        def create_int_feature(values):
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return feature

        features = collections.OrderedDict()
        features["unique_ids"] = create_int_feature([feature.unique_id])
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)

        if self.is_training:
            features["start_positions"] = create_int_feature([feature.start_position])
            features["end_positions"] = create_int_feature([feature.end_position])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        self._writer.write(tf_example.SerializeToString())

    def close(self):
        self._writer.close()


def validate_flags_or_throw(bert_config):
    """Validate the input FLAGS or throw an exception."""
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if FLAGS.do_train:
        if not FLAGS.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if FLAGS.do_predict:
        if not FLAGS.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT_88.5 model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
        raise ValueError(
            "The max_seq_length (%d) must be greater than max_query_length "
            "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    # validate_flags_or_throw(bert_config)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config = run_config.replace(session_config=session_config)
    run_config = run_config.replace(keep_checkpoint_max=2)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = read_squad_examples(
            input_file="filter_data/train_data.csv", is_training=True)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        # Pre-shuffle the input to avoid having to make a very large shuffle
        # buffer in in the `input_fn`.
        rng = random.Random(12345)
        rng.shuffle(train_examples)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        # We write to a temporary file to avoid storing very large constant tensors
        # in memory.
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
            is_training=True)
        convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=train_writer.process_feature)
        train_writer.close()

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num orig examples = %d", len(train_examples))
        tf.logging.info("  Num split examples = %d", train_writer.num_features)
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        del train_examples

        eval_examples = read_squad_examples(
            input_file="filter_data/dev_data.csv", is_training=True)

        eval_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
            is_training=True)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=True,
            output_fn=append_feature)
        eval_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        train_input_fn = input_fn_builder(
            input_file=train_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps,
                        hooks=[EvalHook(estimator,
                                        eval_writer.filename,
                                        eval_features,
                                        eval_steps=FLAGS.save_checkpoints_steps,
                                        max_seq_length=FLAGS.max_seq_length,
                                        max_answer_length=FLAGS.max_answer_length,
                                        checkpoint_dir="SAVE_MODEL",
                                        input_fn_builder=input_fn_builder,
                                        th=86,
                                        model_name="BERT")])

    if FLAGS.do_predict:
        test_examples = read_squad_examples(
            input_file="filter_data/test_data.csv", is_training=False)

        test_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "test.tf_record"),
            is_training=False)
        test_features = []

        def append_feature(feature):
            test_features.append(feature)
            test_writer.process_feature(feature)

        convert_examples_to_features(
            examples=test_examples,
            tokenizer=tokenizer,
            max_seq_length=FLAGS.max_seq_length,
            doc_stride=FLAGS.doc_stride,
            max_query_length=FLAGS.max_query_length,
            is_training=False,
            output_fn=append_feature)
        test_writer.close()

        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(test_examples))
        tf.logging.info("  Num split examples = %d", len(test_features))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=test_writer.filename,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # If running eval on the TPU, you will need to specify the number of
        # steps.
        predictions = estimator.predict(predict_input_fn, yield_single_examples=True)

        #             predictions = {
        #                 "unique_ids": unique_ids,
        #                 "start_logits": start_logits,
        #                 "end_logits": end_logits,
        #             }

        instances = []
        with open("results.csv", "w", encoding="utf-8") as fw:
            # for qa_id in hoka_test:
            #     value = "NaN"
            #     fw.write(f"\"{qa_id}\",\"{value}\"\n")
            for i, item in enumerate(predictions):
                unique_ids = item["unique_ids"]
                qa_id = test_features[i].unique_id
                # print(unique_ids, type(unique_ids))
                # print(qa_id, type(qa_id))
                assert qa_id == unique_ids

                start_logits = item["start_logits"]
                end_logits = item["end_logits"]

                n_best_item = write_prediction(test_features[i], start_logits, end_logits,
                                               n_best_size=20, max_answer_length=FLAGS.max_answer_length)

                best_list = [a["text"] for a in n_best_item[:3]]

                fw.write("\"{}\",\"{}\",\"{}\",\"{}\"\n".format(qa_id, *best_list))

                # yp1 = item["yp1"]
                # yp2 = item["yp2"]
                #
                # input_ids = test_features[i].input_ids
                # pred_ids = input_ids[yp1:yp2+1]
                # pred_text = tokenizer.convert_ids_to_tokens(pred_ids)
                #
                # tok_text = " ".join(pred_text)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")
                #
                # # Clean whitespace
                # tok_text = tok_text.strip()
                # tok_text = " ".join(tok_text.split())
                # final_text = " "
                # for c in tok_text.split():
                #     if (("a" <= c[0] <= "z") or ("A" <= c[0] <= "Z")) and \
                #        (("a" <= final_text[-1] <= "z") or ("A" <= final_text[-1] <= "Z")):
                #         final_text += " " + c
                #     else:
                #         final_text += c
                # final_text = final_text.strip()
                # fw.write(f"\"{qa_id}\",\"{final_text}\"\n")
                # instances.append((qa_id, yp1, yp2, y1, y2))


if __name__ == "__main__":
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()
