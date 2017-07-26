"""
多层卷积层加池化层的深度卷积神经网络

另一种网络构建方式见：mnist_dnn.py
"""

import tensorflow as tf

def model(top_k, out_units):
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    labels = tf.placeholder(dtype=tf.int32)
    training_or_not = tf.placeholder(dtype=tf.bool, shape=[])

    l2_reular_scale = 0.0005

    # 核的默认初始化方式貌似是xavier
    conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='conv1zzz')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')

    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv3_2 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding='same',
                               activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2, padding='same')

    conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv4_2 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3, 3], padding='same',
                               activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dropout4 = tf.layers.dropout(inputs=conv4_2, rate=0, training=training_or_not)
    pool4 = tf.layers.max_pooling2d(inputs=dropout4, pool_size=[2, 2], strides=2, padding='same')

    flat = tf.reshape(pool4, [-1, 4 * 4 * 512])
    dense1 = tf.layers.dense(inputs=flat, units=4096, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reular_scale))
    dropout5 = tf.layers.dropout(inputs=dense1, rate=0.5, training=training_or_not)

    y_result = tf.layers.dense(inputs=dropout5, units=out_units,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

    # loss and accuracy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y_result))
    labels_f = tf.cast(labels, tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_result, 1), labels_f), tf.float32))

    # top k accuracy and out
    probabilities = tf.nn.softmax(y_result)
    accuracy_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    val_top_k, index_top_k = tf.nn.top_k(probabilities, k=top_k)

    # 优化方法
    lr = tf.Variable(0.0004, dtype=tf.float32, trainable=False)
    train_method = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('accuracy_top_k', accuracy_top_k)
    merged_summary_op = tf.summary.merge_all()

    return {'images': images,
            'labels': labels,
            'training_or_not': training_or_not,
            'train_method': train_method,
            'learning_rate': lr,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_top_k,
            'val_top_k': val_top_k,
            'index_top_k': index_top_k,
            'merged_summary_op': merged_summary_op}


def model2(top_k, out_units):
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
    labels = tf.placeholder(dtype=tf.int32)
    training_or_not = tf.placeholder(dtype=tf.bool, shape=[])


    conv1 = tf.layers.conv2d(inputs=images, filters=64, kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             name='conv1zzz')
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

    conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='same')

    conv3 = tf.layers.conv2d(inputs=pool2, filters=256, kernel_size=[3,3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv3_2 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding='same',
                               activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    pool3 = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2, padding='same')

    conv4 = tf.layers.conv2d(inputs=pool3, filters=512, kernel_size=[3, 3], padding='same',
                             activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    conv4_2 = tf.layers.conv2d(inputs=conv4, filters=512, kernel_size=[3, 3], padding='same',
                               activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    dropout4 = tf.layers.dropout(inputs=conv4_2, rate=0, training=training_or_not)
    pool4 = tf.layers.max_pooling2d(inputs=dropout4, pool_size=[2, 2], strides=2, padding='same')

    flat1 = tf.reshape(pool4, [-1, 4 * 4 * 512])
    flat2 = tf.reshape(pool2, [-1, 16 * 16 * 128])
    flat = tf.concat([flat1, flat2], 1)
    dense1 = tf.layers.dense(inputs=flat, units=4096, activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
    dropout5 = tf.layers.dropout(inputs=dense1, rate=0.5, training=training_or_not)

    y_result = tf.layers.dense(inputs=dropout5, units=out_units,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

    # loss and accuracy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y_result))
    labels_f = tf.cast(labels, tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_result, 1), labels_f), tf.float32))

    # top k accuracy and out
    probabilities = tf.nn.softmax(y_result)
    accuracy_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    val_top_k, index_top_k = tf.nn.top_k(probabilities, k=top_k)

    # 优化方法
    lr = tf.Variable(0.0004, dtype=tf.float32, trainable=False)
    train_method = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('accuracy_top_k', accuracy_top_k)
    merged_summary_op = tf.summary.merge_all()

    return {'images': images,
            'labels': labels,
            'training_or_not': training_or_not,
            'train_method': train_method,
            'learning_rate': lr,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_top_k,
            'val_top_k': val_top_k,
            'index_top_k': index_top_k,
            'merged_summary_op': merged_summary_op}
