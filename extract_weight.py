import tensorflow as tf
import pickle
import cnn_model

MODEL_PATH = "./model"

def extract_weight():
    model = cnn_model.model(5, 3755)

    # 初始化
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(MODEL_PATH)
    saver.restore(sess, ckpt)


    v = tf.all_variables()
    d = {}
    for i in range(16):
        a = v[i]
        print(a.name)
        value = sess.run(a)
        d[a.name] = value

    with open("test.bin", 'wb') as f:
        pickle.dump(d, f)

    sess.close()


if __name__ == '__main__':
    extract_weight()