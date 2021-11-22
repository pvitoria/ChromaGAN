from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
import glob
import os.path
import config

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

class VideoDataLoader():
    @pipeline_def
    def video_pipe(filenames, sequence_length, initial_prefetch_size, stride, video_dim):
        videos = fn.readers.video_resize(
            device="gpu", 
            filenames=filenames, 
            sequence_length=sequence_length, 
            initial_fill=initial_prefetch_size, 
            image_type=types.DALIImageType.RGB,
            random_shuffle=True,
            stride=stride,
            resize_y=video_dim,
            resize_x=video_dim,
            name="Reader",
            dtype=types.DALIDataType.UINT8,
            file_list_include_preceding_frame=True
        )

#         videos_cropped = fn.crop(
#             videos, 
#             device="gpu",
#             crop_w=video_dim,
#             crop_h=video_dim,
#             crop_pos_x=0.5, 
#             crop_pos_y=0.5
#         )
        
        videos_lab = fn.color_space_conversion(
            videos,
            device="gpu",
            image_type=types.DALIImageType.RGB,
            output_type=types.DALIImageType.Lab,
        )

        return videos_lab

    def __init__(self, data_dir, image_size=224, batch_size=8, sequence_length=8, stride=1):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.video_dim = image_size
        
        files = glob.glob(os.path.join(config.DATA_DIR, data_dir, "*.mp4"))
        
        self.pipe = VideoDataLoader.video_pipe(
            batch_size=batch_size, 
            sequence_length=sequence_length, 
            stride=stride,
            video_dim=image_size, 
            num_threads=6, 
            device_id=0, 
            filenames=files,
            seed=123456,
            initial_prefetch_size=256
        )
        self.pipe.build()
        meta = self.pipe.reader_meta("Reader")
        self.size = meta["epoch_size_padded"] // meta["number_of_shards"]
        self.daliop = dali_tf.DALIIterator()
    
    def generate_batch(self):
        image_raw = self.daliop(
            pipeline = self.pipe,
            shapes = [(self.batch_size, self.sequence_length, self.video_dim, self.video_dim, 3)],
            dtypes = [tf.uint8])[0]
        image = tf.cast(tf.reshape(image_raw, (self.batch_size * self.sequence_length, self.video_dim, self.video_dim, 3)), tf.float32) / 255.0
        image_l = image[:, :, :, 0:1]
        image_ab = image[:, :, :, 1:]
        image_l3 = tf.tile(image_l, (1, 1, 1, 3))
        
        return image_l, image_ab, image_l3

