import tensorflow as tf
#split 20%testset 80%dataset
def split_train_test_list(a_list):
    return a_list[:int(len(a_list)*0.2)], a_list[int(len(a_list)*0.2):]

#process image as tf format    
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range
    return image
    
def test_image(path):
    test_tensor = load_and_preprocess_image(path)
    test_tensor = tf.expand_dims(test_tensor, axis=0)
    return model.predict(test_tensor)