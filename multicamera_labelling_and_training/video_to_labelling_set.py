from pathlib import Path
import numpy as np
import yaml
import copy
from datetime import datetime
import pandas as pd
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import logging
import shutil
import tempfile
import os

logger = logging.getLogger(__name__)


def represent_none(self, _):
    """format yaml to represent None as ~"""
    return self.represent_scalar("tag:yaml.org,2002:null", "~")


yaml.add_representer(type(None), represent_none)


class LabellingSetCreator:
    def __init__(
        self,
        recording_directory,
        output_directory,
        skeleton_dict,
        cameras,
        n_individuals=1,
        skeleton_name="RodentBody",
        n_jobs=5,
        calibration_directory=None,
        remux_videos=False,
    ):
        self.recording_directory = Path(recording_directory)
        if calibration_directory is not None:
            self.calibration_directory = Path(calibration_directory)
        else:
            self.calibration_directory = None
        self.output_directory = Path(output_directory)
        self.skeleton_dict = skeleton_dict
        self.cameras = cameras
        self.n_individuals = n_individuals
        self.skeleton_name = skeleton_name
        self.n_jobs = n_jobs
        self.video_df = None
        self.total_frames = None
        assert self.n_individuals > 0, "n_individuals must be greater than 0"
        self.remux_videos = remux_videos

    def detect_videos(self, stem="mp4"):
        videos = list(self.recording_directory.glob(f"*.{stem}"))
        self.video_df, self.total_frames = get_video_frame_df(
            self.recording_directory, videos, self.cameras
        )
        # ensure the video dataframe is sorted
        self.video_df.sort_values(by="frame_start", ascending=True)

    def do_remux(self):
        """Remux videos to ensure they are properly muxxed, creates remuxxed
        videos in a temp directory, then deletes that directory after create_labelling_set"""

        # create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        # remux videos
        for video in tqdm(self.video_df.video_location, desc="remuxing videos"):
            output_video = temp_dir / video.name
            remux_command = f"ffmpeg -i {video} -c copy -y {output_video}"
            os.system(remux_command)
        # update the video dataframe
        self.video_df.video_location = [
            temp_dir / video.name for video in self.video_df.video_location
        ]

    def create_labelling_set(self, dataset_name, n_frames_to_grab=20):
        logger.info(f"Creating labelling set for {dataset_name} at {self.output_directory}")
        # find the videos
        if self.video_df is None:
            self.detect_videos()
        logger.info(f"Found {len(self.video_df)} videos")
        # create the output directory
        self.output_directory.mkdir(exist_ok=True)

        # remux videos if necessary
        if self.remux_videos:
            self.do_remux()

        # generate a yaml dictionary
        recording_name = self.recording_directory.stem
        yaml_dict = self.skeleton_dict
        yaml_dict["Name"] = recording_name
        yaml_dict["Recordings"][recording_name] = None
        yaml_dict["Date of creation"] = datetime.now().date()
        yaml_dict["Cameras"] = [str(cam) for cam in self.cameras]

        # save the yaml
        yaml_path = self.output_directory / dataset_name / f"{dataset_name}.yaml"
        # ensure the directory
        (self.output_directory / dataset_name).mkdir(parents=True, exist_ok=True)
        # Use yaml.dump() to write the dictionary to a YAML file
        with open(yaml_path, "w") as yaml_file:
            yaml.dump(yaml_dict, yaml_file, default_flow_style=False, default_style=None)

        # create a pandas dataframe for annotations.csv
        bodyparts = yaml_dict["Keypoints"]

        # create a labeling file for each individual
        scorers_row = list(np.tile(["Scorer"], len(bodyparts) * 3 * self.n_individuals))
        entities_row = np.concatenate(
            [
                np.tile([self.skeleton_name + str(i)], len(bodyparts) * 3)
                for i in range(self.n_individuals)
            ]
        )
        bodyparts_row = np.concatenate(
            [np.repeat(bodyparts, 3) for i in range(self.n_individuals)]
        )
        coords_row = np.concatenate(
            [np.tile(["x", "y", "state"], len(bodyparts)) for i in range(self.n_individuals)]
        )
        # Create the multi-index header
        # print(len(scorers_row), len(entities_row), len(bodyparts_row),len(coords_row))
        arrays = [scorers_row, entities_row, bodyparts_row, coords_row]
        header = pd.MultiIndex.from_arrays(arrays)

        # Create a DataFrame with the given header
        labels_df = pd.DataFrame(columns=header)
        labels_df.columns.names = ["Scorer", "entities", "bodyparts", "coords"]

        # randomly choose which frames to grab
        frames_to_grab = np.sort(
            np.random.choice(self.total_frames, n_frames_to_grab, replace=False)
        )

        # for each frame to grab, get the frame for each camera,
        #   and create a row in the csv
        for frame_num in frames_to_grab:
            ### add row to labels_df
            labels_df.loc[f"Frame_{frame_num}.jpg"] = list(
                np.tile(("", "", "0"), len(bodyparts) * self.n_individuals)
            )

        # create the annotations file
        for camera in self.cameras:
            annotation_file = (
                self.output_directory / dataset_name / recording_name / camera / f"annotations.csv"
            )

            # ensure the directory exists
            annotation_file.parent.mkdir(parents=True, exist_ok=True)

            labels_df.to_csv(annotation_file)
            # this is just a formatting peculiarity of jarvis
            append_comma_to_csv_from_line_n(annotation_file)

        logger.info("Determine which frames to grab")
        # create a list of [camera, frame number, video, output_location]
        frame_list = []
        for frame_num in frames_to_grab:
            for camera in self.cameras:
                # the video index where this frame is located
                video_idx = np.where(
                    frame_num > self.video_df[self.video_df.camera == camera].frame_start.values
                )[0][-1]
                # which frame to grab from the video
                frame_to_grab = (
                    frame_num
                    - self.video_df[self.video_df.camera == camera].iloc[video_idx].frame_start
                )
                # path of the video
                video_location = (
                    self.video_df[self.video_df.camera == camera].iloc[video_idx].video_location
                )

                # write file
                output_filename = (
                    self.output_directory
                    / dataset_name
                    / recording_name
                    / camera
                    / f"Frame_{frame_num}.jpg"
                )
                frame_list.append((camera, frame_to_grab, video_location, output_filename))
        frame_to_grab_df = pd.DataFrame(
            frame_list, columns=["camera", "frame_to_grab", "video_location", "output_filename"]
        )
        logger.info("Grabbing images from video")
        # save frames
        Parallel(n_jobs=self.n_jobs)(
            delayed(process_frame)(row.video_location, row.frame_to_grab, row.output_filename)
            for idx, row in tqdm(
                frame_to_grab_df.iterrows(),
                total=len(frame_to_grab_df),
                desc="grabbing images from video",
            )
        )
        logger.info("moving calibration directory")
        if self.calibration_directory is not None:
            if (self.output_directory / dataset_name).exists() == False:
                # copy the calibration file
                shutil.copytree(
                    self.calibration_directory,
                    self.output_directory / dataset_name,
                )
            else:
                logger.info("Calibration folder already exists")


def process_frame(video_location, frame_to_grab, output_filename):
    """Grab a frame from a video and save it as a jpg"""
    output_filename = Path(output_filename)
    video_location = Path(video_location)
    cap = cv2.VideoCapture(video_location.as_posix())
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_grab)
    ret, img = cap.read()
    assert ret, f"Could not read frame {frame_to_grab} from {video_location}"
    cap.release()
    cv2.imwrite(output_filename.as_posix(), img)


def append_comma_to_csv_from_line_n(filename, n=4):
    """Append a comma to the end of each line starting from line n"""
    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w") as file:
        for i, line in enumerate(lines):
            if i >= n - 1:
                line = line.strip() + ",\n"
            file.write(line)


def get_video_frame_df(recording_directory, videos, cameras):
    """This is a simple custom frunction to create a dataframe of
    videos and their corresponding cameras and frame times from my data structure
    """
    # count the number of frames for each video
    n_frames = {}
    for camera in cameras:
        n_frames[camera] = sum(
            [
                len(pd.read_csv(i))
                for i in list(recording_directory.glob(f"*{camera}.*.metadata.csv"))
            ]
        )
        assert n_frames[camera] > 0
    n_frames_array = np.array(list(n_frames.values()))
    total_frames = n_frames_array[0]
    assert np.all(n_frames_array == total_frames)

    # create dataframe of videos
    video_df = pd.DataFrame(columns=["camera", "frame_start", "video_location"])
    for video in videos:
        if np.any([cam in video.stem for cam in cameras]) == False:
            continue
        if (video.stem.count(".")) == 1:
            sn, cam = video.stem.split(".")
            frame = 0
        elif (video.stem.count(".")) == 2:
            dt, cam, frame = video.stem.split(".")
            frame = int(frame)
        else:
            raise ValueError
        video_df.loc[len(video_df)] = [cam, frame, video]
    video_df = video_df.sort_values(by="frame_start", ascending=True)
    return video_df, total_frames
