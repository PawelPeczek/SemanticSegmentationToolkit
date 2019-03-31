import tensorflow as tf


class EvaluationUtils:

    @staticmethod
    def evaluate_miou(sess: tf.Session, mean_iou: tf.Operation, mean_iou_update: tf.Operation) -> float:
        try:
            while True:
                sess.run(mean_iou_update)
        except tf.errors.OutOfRangeError:
            return sess.run(mean_iou)
