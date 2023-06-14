import onnx
from onnx2keras import onnx_to_keras
import tensorflow as tf
from tensorflow.python.keras import backend as K


sess = tf.compat.v1.Session()
K.set_session(sess)

#onnx_model = onnx.load(f'models/mb1-ssd.onnx')
onnx_model = onnx.load(f'models/mb2-ssd-lite.onnx')

input_names = ['image_array']
# change_ordering=True で NCHW形式のモデルをNHWC形式のモデルに変換できる
k_model = onnx_to_keras(onnx_model=onnx_model, input_names=input_names ,change_ordering=True, verbose=True)

# 後々weightを再ロードするためにとっておく
weights = k_model.get_weights()

# saved_model.pbへと保存する
K.set_learning_phase(0)
with K.get_session() as sess:
    # FailedPreconditionErrorを回避するために必要
    init = tf.global_variables_initializer()
    sess.run(init)

    # このままだとweightの情報が消えているのでweightを再ロード
    k_model.set_weights(weights)

    tf.saved_model.simple_save(
        sess,
        str(saved_model_dir.joinpath('1')),
        inputs={'image_array': k_model.input},
        outputs={'category_id': k_model.output}
    )
