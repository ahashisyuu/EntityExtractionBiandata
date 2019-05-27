import collections
import math
import os
import numpy as np
import pandas as pd

from tensorflow.python.training.training_util import _get_or_create_global_step_read as get_global_step
from tensorflow.python.platform import tf_logging as logging
from postprocessing import process
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook


class EvalHook(SessionRunHook):
    def __init__(self,
                 estimator,
                 dev_file,
                 eval_features,
                 eval_steps=100,
                 max_seq_length=300,
                 max_answer_length=15,
                 checkpoint_dir=None,
                 input_fn_builder=None,
                 th=86,
                 model_name=None):
        self.estimator = estimator
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.dev_file = dev_file
        self.eval_features = eval_features
        self.th = th
        self.checkpoint_dir = checkpoint_dir
        self.org_dir = "TRAIN_" + model_name
        if os.path.exists("./EVAL_LOG") is False:
            os.mkdir("./EVAL_LOG")

        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        self._log_save_path = os.path.join("./EVAL_LOG", model_name)
        self.save_path = os.path.join(self.checkpoint_dir, model_name)
        if os.path.exists(self.save_path) is False:
            os.mkdir(self.save_path)

        self._timer = SecondOrStepTimer(every_steps=eval_steps)
        self._steps_per_run = 1
        self._global_step_tensor = None

        self.input_fn_builder = input_fn_builder

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        # self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
        self._global_step_tensor = get_global_step()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use CheckpointSaverHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                metrics = self.evaluation(global_step)
                # print("================", MAP, MRR, self.th, type(MAP), type(MRR), type(self.th))
                if metrics["acc"] > self.th:
                    # print("================", MAP, MRR)
                    self._save(run_context.session, global_step, metrics)

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            metrics = self.evaluation(last_step)
            if metrics["acc"] > self.th:
                self._save(session, last_step, metrics)

    def evaluation(self, global_step):
        print("======================================================")
        print("EVAL STARTING !!!!\n")

        dev_input_fn = self.input_fn_builder(
            input_file=self.dev_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        predictions = self.estimator.predict(dev_input_fn, yield_single_examples=True)

        #             predictions = {
        #                 "unique_ids": unique_ids,
        #                 "start_logits": start_logits,
        #                 "end_logits": end_logits,
        #             }

        with open("./SAVE_MODEL/temp_results.csv", "w", encoding="utf-8") as fw:
            for i, item in enumerate(predictions):
                unique_ids = item["unique_ids"]
                qa_id = self.eval_features[i].unique_id
                # print(unique_ids, type(unique_ids))
                # print(qa_id, type(qa_id))
                assert qa_id == unique_ids

                start_logits = item["start_logits"]
                end_logits = item["end_logits"]
                # yp1 = item["yp1"]
                # yp2 = item["yp2"]
                #
                # y1 = self.eval_features[i].start_position
                # y2 = self.eval_features[i].end_position

                n_best_items = write_prediction(self.eval_features[i], start_logits, end_logits,
                                                n_best_size=20, max_answer_length=self.max_answer_length)
                best_list = [a["text"] for a in n_best_items[:3]]

                while len(best_list) < 3:
                    best_list.append("empty")

                fw.write("\"{}\",\"{}\",\"{}\",\"{}\"\n".format(qa_id, *best_list))
                # instances.append((qa_id, yp1, yp2, y1, y2))

        dev_data = pd.read_csv("./filter_data/dev_data.csv",
                                header=None,
                                names=["id", "sent", "entity", "label"])
        results_data = pd.read_csv("./SAVE_MODEL/temp_results.csv",
                                   header=None,
                                   names=["id", "s1", "s2", "s3"])
        results_data["sent"] = dev_data["sent"]
        results_data["entity"] = dev_data["entity"]
        results_data["label"] = dev_data["label"]
        results_data["final"] = results_data.apply(process, axis=1)
        final_results = results_data[["id", "final", "label"]]
        final_results["EM"] = final_results.apply(is_equal, axis=1)

        EM = final_results["EM"].to_numpy(dtype=np.int)
        acc = np.sum(EM) / EM.shape[0]
        metrics = {'global_step': global_step, "acc": acc}
        print(f"golbal_step: {global_step}, acc: {acc}")
        return metrics

    def _save(self, session, step, metrics=None):
        """Saves the latest checkpoint, returns should_stop."""
        save_path = os.path.join(self._save_path, "step{}_acc{:5.4f}".format(step, metrics["acc"]))

        list_name = os.listdir(self.org_dir)
        for name in list_name:
            if "model.ckpt-{}".format(step-1) in name:
                org_name = os.path.join(self.org_dir, name)
                tag_name = save_path + "." + name.split(".")[-1]
                print("save {} to {}".format(org_name, tag_name))
                with open(org_name, "rb") as fr, open(tag_name, 'wb') as fw:
                    fw.write(fr.read())


def is_equal(x):
    index, final, label = x
    return 1 if final == label else 0


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"])


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


def write_prediction(feature, start_logits, end_logits, n_best_size, max_answer_length):
    start_indexes = _get_best_indexes(start_logits, n_best_size)
    end_indexes = _get_best_indexes(end_logits, n_best_size)

    tokens = feature.tokens
    mini_index = tokens.index("[SEP]")

    prelim_predictions = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature.tokens):
                continue
            if end_index >= len(feature.tokens):
                continue
            if start_index <= mini_index:
                continue
            if end_index <= mini_index:
                continue
            # if not feature.token_is_max_context.get(start_index, False):
            #     continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=feature.unique_id,
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index]))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break

        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        final_text = " "
        for c in tok_text.split():
            if (("a" <= c[0] <= "z") or ("A" <= c[0] <= "Z")) and \
                    (("a" <= final_text[-1] <= "z") or ("A" <= final_text[-1] <= "Z")):
                final_text += " " + c
            else:
                final_text += c
        final_text = final_text.strip()

        if final_text in seen_predictions:
            continue
        if len(final_text) <= 1:
            continue

        def contain_punct(_text):
            if "、" in _text or "," in _text or \
               "." in _text or "，" in _text or \
               "。" in _text or ":" in _text or \
               "：" in _text or "%" in _text:
                return True
            return False

        if contain_punct(final_text) is True:
            continue

        seen_predictions[final_text] = True
        nbest.append(
            _NbestPrediction(
                text=final_text,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    if not nbest:
        nbest.append(
            _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
        if not best_non_null_entry:
            if entry.text:
                best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["probability"] = probs[i]
        output["start_logit"] = entry.start_logit
        output["end_logit"] = entry.end_logit
        nbest_json.append(output)

    assert len(nbest_json) >= 1

    return nbest_json




