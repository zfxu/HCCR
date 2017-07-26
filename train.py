import tensorflow as tf
import numpy as np
import os
import cnn_model
import time
import cv2
import pickle
from make_tf_data import pre_process

tf.app.flags.DEFINE_integer('charset_size', 200, "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('label_size', 4, "Length of one label.")
tf.app.flags.DEFINE_integer('batch_size', 500, "Num of each batch.")
tf.app.flags.DEFINE_integer('val_batch_size', 1000, "Num of each validation batch.")
tf.app.flags.DEFINE_integer('epoch_num', 50, "Num of epochs.")
tf.app.flags.DEFINE_integer('sample_num', 897758, "Num of samples.")
tf.app.flags.DEFINE_string('train_data_dir', './train_200_inv/', "Train data dir while images in")
tf.app.flags.DEFINE_string('val_data_dir', './test_200_inv/', "Validation data dir while images in")
tf.app.flags.DEFINE_string('model_path', './model/', "Path the model in")
tf.app.flags.DEFINE_string('model_name', 'model.ckpt', "Model name")
tf.app.flags.DEFINE_bool('restore', False, "If restore model from file")
tf.app.flags.DEFINE_string('test_pic', './15.png', "Test picture.")
tf.app.flags.DEFINE_string('char_dict', './char_dict.bin', "Char dict.")
tf.app.flags.DEFINE_string('log_dir', './log_dir', "Tf summary log.")
tf.app.flags.DEFINE_integer('run_type', 0, "0 for traning;1 for validation;2 for inference")

FLAGS = tf.app.flags.FLAGS


# reading tf data
class DataReader:
    def __init__(self, data_dir, bath_size, num_epochs, is_training):
        l = os.listdir(data_dir)
        l = map(lambda x: os.path.join(data_dir, x), l)
        self.file_names = list(l)
        self.batch_size = bath_size
        self.num_epochs = num_epochs
        self.is_training = is_training

    def input(self):
        file_queue = tf.train.string_input_producer(self.file_names, num_epochs=self.num_epochs)
        image, label = self.read_and_decode(file_queue)

        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=self.batch_size,
                                                                  capacity=5000, min_after_dequeue=1000)
        return image_batch, tf.reshape(label_batch, [self.batch_size])

    def read_and_decode(self, file_queue):
        image_bytes = FLAGS.image_size * FLAGS.image_size
        label_bytes = FLAGS.label_size
        record_bytes = image_bytes + label_bytes

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # 提取标签
        label = tf.cast(features['label'], tf.int32)

        # 提取图像
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [1, FLAGS.image_size, FLAGS.image_size])  # depth, height, width
        # Convert from [depth, height, width] to [height, width, depth].
        image = tf.transpose(image, [1, 2, 0])

        image = self.preprocess(image)

        return image, label

    def preprocess(self, image):
        # images = tf.image.random_flip_up_down(images)
        # images = tf.image.random_brightness(images, max_delta=0.3)
        # images = tf.image.random_contrast(images, 0.8, 1.2)
        def rotate_image(image):
            angle = tf.random_uniform([], minval=-20*2*np.pi/360, maxval=20*2*np.pi/360)
            # angle = tf.truncated_normal([], mean=0.0, stddev= 15*2*np.pi/360/3)
            image = tf.contrib.image.rotate(image, angle)
            return image

        def scale_image(image):
            scale_x = tf.random_uniform([], minval=0.8, maxval=1.2)
            scale_y = tf.random_uniform([], minval=0.8, maxval=1.2)
            size_x = tf.cast(tf.multiply(scale_x, tf.cast(image.shape[0], dtype='float32')), dtype='int32')
            size_y = tf.cast(tf.multiply(scale_y, tf.cast(image.shape[1], dtype='float32')), dtype='int32')
            image = tf.image.resize_images(image, [size_x, size_y])
            image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.image_size, FLAGS.image_size)
            return image

        image = tf.cast(image, 'float32')

        if self.is_training:
            image = scale_image(image)
            image = rotate_image(image)

        image = image * (1. / 255) - 0.5

        return image

# class DataReader:
#     def __init__(self, data_dir, bath_size, num_epochs):
#         self.batch_size = bath_size
#         self.num_epochs = num_epochs
#
#         # 获取所有图像的名字和标签
#         truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
#         print(truncate_path)
#         image_names = []
#         for root, sub_folder, file_list in os.walk(data_dir):
#             if root < truncate_path:
#                 image_names += [os.path.join(root, file_path) for file_path in file_list]
#         random.shuffle(image_names)
#         labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in image_names]
#
#         self.images_tensor = tf.convert_to_tensor(image_names, dtype=tf.string)
#         self.labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int64)
#
#     def input(self):
#         input_queue = tf.train.slice_input_producer([self.images_tensor, self.labels_tensor],
#                                                     num_epochs=self.num_epochs)
#
#         # decode labels and images
#         labels = input_queue[1]
#         images = tf.read_file(input_queue[0])
#         images = tf.image.convert_image_dtype(tf.image.decode_png(images, channels=1), tf.float32)
#
#         images = self.preprocess(images)
#
#         image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=self.batch_size,
#                                                           capacity=5000, min_after_dequeue=100)
#
#         return image_batch, label_batch
#
#     def preprocess(self, images):
#         # images = tf.image.random_flip_up_down(images)
#         images = tf.image.random_brightness(images, max_delta=0.3)
#         images = tf.image.random_contrast(images, 0.8, 1.2)
#
#         new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
#         images = tf.image.resize_images(images, new_size)
#         return images


def train():
    # train data batch
    train_data_reader = DataReader(FLAGS.train_data_dir, FLAGS.batch_size, FLAGS.epoch_num, True)
    image_batch, label_batch = train_data_reader.input()

    # validation data batch
    val_data_reader = DataReader(FLAGS.val_data_dir, FLAGS.val_batch_size, None, False)
    val_image_batch, val_label_batch = val_data_reader.input()

    model = cnn_model.model2(3, FLAGS.charset_size)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # 初始化
    sess = tf.Session()
    sess.run(init_op)

    # name = [v.name for v in tf.all_variables()]

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # modele saver
    saver = tf.train.Saver()

    # summary writer
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')

    step = 0
    if FLAGS.restore:
        ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))
            step += int(ckpt.split('-')[-1]) + 1

            lr = sess.run(tf.assign(model['learning_rate'], 0.0001))
            print(lr)

    try:
        while not coord.should_stop():
            start_time = time.time()

            # 图像和标签数据
            train_images_batch, train_labels_batch = sess.run([image_batch, label_batch])
            feed_dict = {model['images']: train_images_batch, model['labels']: train_labels_batch}


            # test
            if step == 0:
                for i in range(len(train_images_batch)):
                    img = train_images_batch[i, :, :, :]
                    img = (img+0.5) * 255
                    img = img.astype('uint8')
                    cv2.imwrite('./temp/'+str(i)+'.bmp', img)

            # 输出准确率
            if step % 10 == 0:
                feed_dict[model['training_or_not']] = False
                ac_res, train_summary = sess.run([model['accuracy'], model['merged_summary_op']], feed_dict=feed_dict)
                print('Step %d: accuracy = %.4f' % (step, ac_res))
                train_writer.add_summary(train_summary, step)

            # step_interval = (FLAGS.sample_num // FLAGS.batch_size)*3
            # if step % step_interval == 0 and step <= 5 * step_interval and step != 0:
            #     sess.run(tf.assign(model['learning_rate'], model['learning_rate']/2))
            #     print('-'*30)
            #     print('learning rate:', sess.run(model['learning_rate']))

            # 反向传播训练
            feed_dict[model['training_or_not']] = True
            sess.run(model['train_method'], feed_dict=feed_dict)

            # 保存模型
            if step % 50 == 0:
                saver.save(sess, FLAGS.model_path + FLAGS.model_name, global_step=step)
                # saver.save(sess, FLAGS.model_path + FLAGS.model_name)

                val_images_batch, val_labels_batch = sess.run([val_image_batch, val_label_batch])
                feed_dict[model['images']] = val_images_batch
                feed_dict[model['labels']] = val_labels_batch
                feed_dict[model['training_or_not']] = False
                val_summary = sess.run(model['merged_summary_op'], feed_dict=feed_dict)
                val_writer.add_summary(val_summary, step)

            # 统计时间
            if step % 10 == 0:
                end_time = time.time()
                print('Time use: %.4f' % (end_time - start_time))

            step += 1

    except tf.errors.OutOfRangeError:
        print('Done training')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    print('finish!')


def validation():
    val_data_reader = DataReader(FLAGS.val_data_dir, 1000, 1, False)
    image_batch, label_batch = val_data_reader.input()

    model = cnn_model.model2(5, FLAGS.charset_size)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # 初始化
    sess = tf.Session()
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # modele saver
    saver = tf.train.Saver()
    # ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
    # saver.restore(sess, ckpt)
    saver.restore(sess, './model/model.ckpt-4550')

    step = 0
    sum1 = 0
    sumk = 0
    try:
        while not coord.should_stop():
            # 图像和标签数据
            val_images_batch, val_labels_batch = sess.run([image_batch, label_batch])
            feed_dict = {model['images']: val_images_batch, model['labels']: val_labels_batch, model['training_or_not']: False}

            accuracy, accuracy_top_k = sess.run([model['accuracy'], model['accuracy_top_k']], feed_dict=feed_dict)
            sum1 += accuracy
            sumk += accuracy_top_k
            step += 1

            if step%50 == 0:
                print('Step %d: accuracy = %.4f, top k accuracy = %.4f' % (step, accuracy, accuracy_top_k))

    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    print('Average: accuracy = %.4f, top k accuracy = %.4f' % (sum1/step, sumk/step))


def inference():
    # build model
    model = cnn_model.model(5, FLAGS.charset_size)

    # 初始化
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
    saver.restore(sess, ckpt)

    # label对应字符
    char_dict = {}
    with open("char_dict2", "rb") as f:
        char_dict = pickle.load(f)

    img = cv2.imread(FLAGS.test_pic, 0)
    img = 255 - img
    img = pre_process(img)
    cv2.imshow("", img)
    cv2.waitKey(0)
    img = np.reshape(img, (1, FLAGS.image_size, FLAGS.image_size, 1))

    feed_dict = {model['images']: img, model['training_or_not']: False}

    out_labels, probs = sess.run([model['index_top_k'], model['val_top_k']], feed_dict=feed_dict)
    print('-'*30)
    print('Top k result')
    for i in range(len(out_labels[0])):
        print('%d: Predicted val %s, Probality %3f' % (i+1, char_dict[out_labels[0][i]], probs[0][i]))

    sess.close()


def validation_test():
    val_data_reader = DataReader(FLAGS.val_data_dir, 1, 1, False)
    image_batch, label_batch = val_data_reader.input()

    model = cnn_model.model(5, FLAGS.charset_size)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # 初始化
    sess = tf.Session()
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # modele saver
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(FLAGS.model_path)
    saver.restore(sess, ckpt)

    # 字典
    char_dict = {}
    with open("char_dict2", "rb") as f:
        char_dict = pickle.load(f)

    # 保存输出结果，错误的情况
    out_file = open('wrong_condition.txt', 'w')

    step = 0
    sum1 = 0
    sumk = 0
    wrong_num = 1
    try:
        while not coord.should_stop():
            # 图像和标签数据
            val_images_batch, val_labels_batch = sess.run([image_batch, label_batch])
            feed_dict = {model['images']: val_images_batch, model['labels']: val_labels_batch, model['training_or_not']: False}

            out_labels, probs = sess.run([model['index_top_k'], model['val_top_k']], feed_dict=feed_dict)

            if out_labels[0][0] != val_labels_batch[0]:
                out_file.write('true:'+char_dict[val_labels_batch[0]]+'\n')
                out_file.write('topk')
                for i in range(len(out_labels[0])):
                    out_file.write(char_dict[out_labels[0][i]]+str(probs[0][i]))
                out_file.write('\n')

                val_images_batch[0] = (val_images_batch[0]+0.5)*255
                cv2.imwrite('./wrong_pic/'+str(wrong_num)+'.bmp', val_images_batch[0])
                wrong_num += 1

            step += 1
            if step%100 == 0:
                print('Step %d' % step)

    except tf.errors.OutOfRangeError:
        print('Done!')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    out_file.close()

    print('Average: accuracy = %.4f, top k accuracy = %.4f' % (sum1/step, sumk/step))


def main(_):
    if FLAGS.run_type == 0:
        train()
    elif FLAGS.run_type == 1:
        validation()
    else:
        inference()


if __name__ == '__main__':
    tf.app.run()

