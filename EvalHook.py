import os
import numpy as np

from tensorflow.python.training.training_util import _get_or_create_global_step_read as get_global_step
from tensorflow.python.platform import tf_logging as logging
from utils import PRF
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook


class EvalHook(SessionRunHook):
    def __init__(self,
                 estimator,
                 dev_file,
                 eval_features,
                 eval_steps=100,
                 max_seq_length=300,
                 checkpoint_dir=None,
                 input_fn_builder=None,
                 th=86,
                 model_name=None):
        self.estimator = estimator
        self.max_seq_length = max_seq_length
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

        instances = []

        for i, item in enumerate(predictions):
            unique_ids = item["unique_ids"]
            qa_id = self.eval_features[i].unique_id
            # print(unique_ids, type(unique_ids))
            # print(qa_id, type(qa_id))
            assert qa_id == unique_ids

            start_logits = item["start_logits"]
            end_logits = item["end_logits"]
            yp1 = item["yp1"]
            yp2 = item["yp2"]

            y1 = self.eval_features[i].start_position
            y2 = self.eval_features[i].end_position

            instances.append((qa_id, yp1, yp2, y1, y2))

        metrics = PRF(instances)

        metrics['global_step'] = global_step
        acc = metrics["acc"]
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
