import tensorflow as tf
import numpy as np


# import librosa
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


####################################################################################
image_small_shape = (250, 250)
number_of_images_small = 100


####################################################################################
def write_signals_to_tfr_short(signals, labels1, labels2, filename: str = "images"):
    filename = filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(filename)
    count = 0
    for index in range(len(signals)):

        signal = signals[index]
        label1 = labels1[index]
        label2 = labels2[index]
        data = {
            'height': _int64_feature(signal.shape[0]),
            'width': _int64_feature(signal.shape[1]),
            'signal': _bytes_feature(serialize_array(signal)),
            'label1': _int64_feature(label1),
            'label2': _int64_feature(label2)
        }

        out = tf.train.Example(features=tf.train.Features(feature=data))
        writer.write(out.SerializeToString())
        count += 1

    writer.close()


####################################################################################

def parse_tfr_element(element):
    data = {'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'signal': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)}
    content = tf.io.parse_single_example(element, data)

    label, signal, height, width = content['label'], content['signal'], content['height'], content['width']

    feature = tf.io.parse_tensor(signal, out_type=tf.int16)
    feature = tf.reshape(feature, shape=[height, width])
    return (feature, label)


def get_dataset_small(filename):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_tfr_element)
    return dataset


if __name__ == "__main__":
    # images_small = np.random.randint(low=0, high=256, size=(number_of_images_small, *image_small_shape), dtype=np.int16)
    # labels_small = np.random.randint(low=0, high=5, size=(number_of_images_small))
    # write_signals_to_tfr_short(images_small, labels_small, filename="small_images")

    dataset_small = get_dataset_small("./small_images.tfrecords")
    for sample in dataset_small.take(10):
        print(sample[0].shape)
        print(sample[1].shape)
    print("Done")
