from fire import Fire

from src.dataset.preprocessing.video import VideoPreprocessor


class VideoGenerator:

    def generate(self, frames_dir_path: str, outputh_path: str, width: int = 2048, height: int = 1024) -> None:
        try:
            video_preprocessor = VideoPreprocessor(frames_dir_path, outputh_path, (width, height))
            video_preprocessor.produce_output_video()
        except Exception as ex:
            print('Failed to produce_output_video. {}'.format(ex))


if __name__ == '__main__':
    Fire(VideoGenerator)
