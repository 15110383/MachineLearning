import preprocessing
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import argparse


class Classifier():
    def __init__(self, scope, img_w, img_h, n_classes, dropout_keep_prob=1.0):

        self.scope = scope
        self.n_classes = n_classes #62
        self.dropout_keep_prob = dropout_keep_prob
        # Giữ chỗ
        self.input = tf.placeholder(tf.float32, [None, img_h, img_w, 1])
        # Convolution layer 1
        self.conv1 = slim.conv2d(
            self.input,
            num_outputs=32, kernel_size=[3, 8],
            stride=[1, 1], padding='Valid',
            scope=self.scope + '_conv1'
        )
        # Convolution layer 2
        self.conv2 = slim.conv2d(
            self.conv1,
            num_outputs=64, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid',
            scope=self.scope + '_conv2'
        )
        # Convolution layer 3
        self.conv3 = slim.conv2d(
            self.conv2,
            num_outputs=128, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid',
            scope=self.scope + '_conv3'
        )
        # Max Pooling (giữ lại pixel có giá trị nhất)
        self.pool = slim.max_pool2d(self.conv3, [2, 2])

        # Chuyển pool thành vector
        self.hidden = slim.fully_connected(
            slim.flatten(self.pool),
            512,
            scope=self.scope + '_hidden',
            activation_fn=tf.nn.relu
        )
        # output là 1 neural 62 phần tử
        self.classes = slim.fully_connected(
            tf.nn.dropout(self.hidden, self.dropout_keep_prob),
            self.n_classes,
            scope=self.scope + '_fc',
            activation_fn=None
        )

        self.targets = tf.placeholder(tf.int32, [None])
        self.targets_onehot = tf.one_hot(self.targets, self.n_classes)
        """
            indices = [0, 1, 2]
            depth = 3
            tf.one_hot(indices, depth)
            [[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]
        """
        # reduce_mean(a) lấy giá trị trung bình của tất cả phần tử
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.targets_onehot,
            logits=self.classes
        ))
        self.train_step = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)


def train(model_name, training_dataset, validation_dataset):
    img_h, img_w = 64, 64
    train_steps = int(1e5)
    batch_size = 10
    # Khởi tạo CNN
    nn = Classifier('classifier', img_w, img_h, len(preprocessing.CLASSES), 0.8)
    # Load training set
    dataset = list(map(lambda f: f.strip(),
                       open(training_dataset, 'r').readlines()))
    # Load validation set
    validation_dataset = list(map(lambda f: f.strip(),
                                  open(validation_dataset, 'r').readlines()))

    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter('summaries/' + model_name)

        for t in range(train_steps):

            # Các bước training
            images, labels = preprocessing.get_batch(dataset, 10, (img_h, img_w))
            loss, _ = sess.run([nn.loss, nn.train_step], feed_dict={
                nn.input: images,
                nn.targets: labels
            })

            # show and save training
            if t % 10 == 0: print(t, loss)
            if t % 1000 == 0: saver.save(sess, 'saves/' + model_name, global_step=t)

            summary = tf.Summary()
            summary.value.add(tag='Loss', simple_value=float(loss))
            if t % 50 == 0:
                # test model với validation set
                images, labels = preprocessing.get_batch(
                    validation_dataset, 20, (img_h, img_w))
                classes = sess.run(nn.classes, feed_dict={nn.input: images})
                summary.value.add(tag='ValidationError',
                                  simple_value=float(sum(np.argmax(classes, -1) != labels)))
            summary_writer.add_summary(summary, t)
            summary_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', type=str, required=True, help='Training dataset name')
    parser.add_argument(
        '-v', type=str, required=True, help='Validation dataset name')
    parser.add_argument('-m', type=str, required=True, help='Model name')

    opt = parser.parse_args()
    train(opt.m, opt.t, opt.v)

    # run training
    # python train.py -t datasplits/good_train -v datasplits/good_validation -m saves/awesome_model