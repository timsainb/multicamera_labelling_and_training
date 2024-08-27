from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
import yaml
import shutil
from tqdm.auto import tqdm
import pandas as pd
import json
import copy
import io
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class TrainingSetCreator:
    def __init__(
        self,
        output_directory,
        trainingset_name,
        percent_validation=0.1,
        keypoints_order=None,
        keypoints_to_ignore=["Tail_Tip", "Tail_Middle", "tail_tip"],
        padding=60,
    ):
        """Creates a COCO formatted training set from labelling sets
        Data Structure:
            - annotations
                - instances_train.json
                - instances_val.json
            - train
                - [dataset name]
                    - [recording name]
                        - [camera name]
                            - [image.jpg]
            - val
                - (same as train)
            - calib_params
                - [dataset name]
                    - {camera}.yaml
        """
        self.output_directory = Path(output_directory) / trainingset_name
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.trainingset_name = trainingset_name
        self.percent_validation = percent_validation
        self.keypoints_order = keypoints_order
        self.keypoints_to_ignore = keypoints_to_ignore
        self.padding = padding # padding around bounding box

    def run(self, labelling_sets):
        dataset_dict, n_keypoints = initialize_jarvis_trainingset(
            self.output_directory, self.keypoints_to_ignore, self.keypoints_order, labelling_sets
        )

        # loop through and add all images
        image_id = 0
        annotation_id = 0
        for ds in labelling_sets:
            dataset_loc = Path(ds["location"])
            logger.info("DATASET:", dataset_loc.name)
            dataset_yaml = list(dataset_loc.glob("*.yaml"))[0]
            use_in_validation = ds["use_in_validation"]
            with open(dataset_yaml, "r") as file:
                dataset_info = yaml.safe_load(file)

            # for each annotation file, create a dataset
            annotations_files = list(dataset_yaml.parent.glob("**/annotations.csv"))

            # copy each image over to the new dataset (for each camera)
            dataset_dict, image_id, annotation_id = copy_images_to_dataset(
                annotations_files,
                dataset_dict,
                dataset_loc,
                self.output_directory,
                image_id,
                annotation_id,
                n_keypoints,
                padding=self.padding
            )

        # create framesets
        frameset_df, n_framesets, dataset_dict = create_framesets(dataset_dict)

        # assign framesets to training and valid, and save training and valid sets
        val_dict, train_dict = assign_framesets_to_train_and_valid(
            n_framesets,
            self.percent_validation,
            labelling_sets,
            frameset_df,
            dataset_dict,
            self.output_directory,
        )

        # save the yaml files
        val_json = self.output_directory / "annotations" / "instances_val.json"
        # Open a YAML file in write mode
        with open(val_json, "w") as file:
            # Write the dictionary to the YAML file
            json.dump(val_dict, file, cls=Int32Encoder)
        train_json = self.output_directory / "annotations" / "instances_train.json"
        # Open a YAML file in write mode
        with open(train_json, "w") as file:
            # Write the dictionary to the YAML file
            json.dump(train_dict, file, cls=Int32Encoder)

        # ensure dataset has properly saved
        with open(train_json, "r") as f:
            val_dict_test = json.load(f)

        logger.info(val_dict_test.keys())


def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def get_row_keypoints(row, keypoint):
    scorer = np.array(row.index)[0][0]
    entity = np.array(row.index)[0][1]
    if keypoint not in row[scorer][entity].index:
        x = 0
        y = 0
        state = 0
    else:
        x = row[scorer][entity][keypoint]["x"]
        y = row[scorer][entity][keypoint]["y"]
        state = row[scorer][entity][keypoint]["state"]
    if np.isnan(x):
        x = 0
    if np.isnan(y):
        y = 0
    if (x == 0) & (y == 0):
        state = 0
    else:
        state = 2
    return np.int32(x), np.int32(y), np.int32(state)


class Int32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def initialize_jarvis_trainingset(save_loc, keypoints_to_ignore, keypoints_order, dataset_locs):
    # create dataset folders
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    Path(save_loc / "annotations").mkdir(parents=True, exist_ok=True)
    Path(save_loc / "train").mkdir(parents=True, exist_ok=True)
    Path(save_loc / "val").mkdir(parents=True, exist_ok=True)

    # this will be populated with all data, then later split into train and validation
    dataset_dict = {}
    dataset_dict["info"] = {
        "contributor": "",
        "date_created": datetime.today().strftime("%Y-%m-%d"),
        "description": "",
        "url": "",
        "versoin": "1.0",
        "year": datetime.now().year,
    }
    # prepopulate the dataset
    dataset_loc = Path(dataset_locs[0]["location"])
    dataset_yaml = list(dataset_loc.glob("*.yaml"))[0]

    with open(dataset_yaml, "r") as file:
        dataset_info = yaml.safe_load(file)

    # remove keypoints_to_ignore
    dataset_info["Keypoints"] = [
        i for i in dataset_info["Keypoints"] if i not in keypoints_to_ignore
    ]

    if keypoints_order is not None:
        dataset_info["Keypoints"] = keypoints_order

    # add keypoints
    dataset_dict["keypoint_names"] = dataset_info["Keypoints"]

    # populate skeleton
    dataset_dict["skeleton"] = []
    if "Skeleton" in dataset_info:
        for key, val in dataset_info["Skeleton"].items():
            if val["Keypoints"][0] in keypoints_to_ignore:
                continue
            if val["Keypoints"][1] in keypoints_to_ignore:
                continue
            dataset_dict["skeleton"].append(
                {
                    "keypointA": val["Keypoints"][0],
                    "keypointB": val["Keypoints"][1],
                    "length": val["Length"][0],
                    "name": key,
                }
            )
    dataset_dict["categories"] = [
        {
            "id": 0,
            "name": "Mouse",
            "num_keypoints": len(dataset_info["Keypoints"]),
            "supercategory": "None",
        }
    ]
    dataset_dict["licenses"] = [{"id": 1, "name": "", "url": ""}]
    dataset_dict["annotations"] = []
    dataset_dict["images"] = []
    n_keypoints = len(dataset_dict["keypoint_names"])

    # add calibration info
    dataset_dict["calibrations"] = {}
    for dataset in dataset_locs:
        calibration_params_loc = list(Path(dataset["location"]).glob("CalibrationParameters"))[0]
        cal_save_loc = save_loc / "calib_params" / Path(dataset["location"]).name
        logger.info("Save locations: ", cal_save_loc, save_loc, "\n\n")
        if cal_save_loc.exists() == False:
            shutil.copytree(calibration_params_loc, cal_save_loc)
        camera_yamls = list(cal_save_loc.glob("*.yaml"))
        dataset_dict["calibrations"][Path(dataset["location"]).name] = {
            camera_yaml.stem: "/".join(
                camera_yaml.parts[np.where(np.array(camera_yaml.parts) == "calib_params")[0][0] :]
            )
            for camera_yaml in camera_yamls
        }
    return dataset_dict, n_keypoints


def copy_images_to_dataset(
    annotations_files,
    dataset_dict,
    dataset_loc,
    save_loc,
    image_id,
    annotation_id,
    n_keypoints,
    recopy_image=False,
    padding=100,
):
    for annotations_file in tqdm(
        annotations_files, desc=f"copying images from {dataset_loc.name}"
    ):
        try:
            annotations = load_jarvis_annotations(annotations_file)
        except Exception as e:
            print(f"Could not load annotations file: {e}")
            continue
        for idx, row in tqdm(
            annotations.iterrows(),
            total=len(annotations),
            leave=False,
            desc=f"{annotations_file.parent.stem}",
        ):
            image_loc = annotations_file.parent / row.name
            width, height = get_image_size(image_loc.as_posix())
            dataset_part = np.where(np.array(image_loc.parts) == dataset_loc.name)[0][0]
            file_name = "/".join(image_loc.parts[dataset_part:])

            image_dict = {
                "coco_url": "",
                "date_captured": "",
                "file_name": file_name,
                "flickr_url": "",
                "height": height,
                "id": image_id,
                "license": 1,
                "width": width,
            }

            dataset_dict["images"].append(image_dict)
            # loop over each entity (each mouse) in the image
            #   TODO: update this so that we can have multiple entity types (e.g. mouse vs pup)
            unique_entities = np.unique(row.index.get_level_values("entities"))
            for entity in unique_entities:
                entity_mask = row.index.get_level_values("entities") == entity

                # get keypoints in format [x, y, state]
                keypoints = [
                    get_row_keypoints(row[entity_mask], keypoint)
                    for keypoint in dataset_dict["keypoint_names"]
                ]
                keypoints_list = list(np.concatenate(keypoints))

                # create a bounding box
                xvals = np.stack(keypoints)[:, 0]
                yvals = np.stack(keypoints)[:, 1]
                xvals = xvals[xvals != 0]
                yvals = yvals[yvals != 0]
                if len(xvals) == 0:
                    continue
                xmin, xmax, ymin, ymax = (
                    np.min(xvals),
                    np.max(xvals),
                    np.min(yvals),
                    np.max(yvals),
                )
                xmin = max([0, xmin-padding])
                ymin = max([0, ymin-padding])
                xmax = min([width, xmax+padding])
                ymax = min([height, ymax+padding])
                
                area = (xmax - xmin) * (ymax - ymin)
                bbox = [
                    float(xmin),
                    float(ymin),
                    float(xmax - xmin),
                    float(ymax - ymin),
                ]
                # create an image dictionary

                annotations_dict = {
                    "area": area,
                    "bbox": bbox,
                    "category_id": 0,
                    "id": annotation_id,
                    "image_id": image_id,
                    "iscrowd": 0,
                    "keypoints": keypoints_list,
                    "num_keypoints": n_keypoints,
                    "segmentation": [],
                }
                annotation_id+=1
                dataset_dict["annotations"].append(annotations_dict)
            image_id += 1
            # copy the image
            save_folder = "train"  #'val' if validation else 'train'
            dst_file = save_loc / save_folder / file_name
            Path(dst_file.parent).mkdir(parents=True, exist_ok=True)
            if (dst_file.exists() == True) & (recopy_image == False):
                continue
            shutil.copy(image_loc, dst_file)
    return dataset_dict, image_id, annotation_id


def create_framesets(dataset_dict):
    ### Create framesets (images that are part of the same frameset)
    # find images that belong to the same frameset,
    frameset_df = pd.DataFrame(
        {
            "frameset": [
                "/".join(Path(i["file_name"]).parts[:-2]) + "/" + Path(i["file_name"]).stem
                for i in dataset_dict["images"]
            ],
            "camera": [Path(i["file_name"]).parts[-2] for i in dataset_dict["images"]],
            "file_id": [i["id"] for i in dataset_dict["images"]],
            "dataset": [Path(i["file_name"]).parts[0] for i in dataset_dict["images"]],
            "path": [Path(i["file_name"]) for i in dataset_dict["images"]],
        }
    )
    framesets = frameset_df.frameset.unique()
    frameset_values = [
        {
            "datasetName": Path(i).parts[0],
            "frames": list(frameset_df[frameset_df.frameset == i].file_id.values),
        }
        for i in framesets
    ]
    dataset_dict["framesets"] = {i: j for i, j in zip(framesets, frameset_values)}
    frameset_df.index = frameset_df["frameset"]
    n_framesets = len(frameset_df.frameset.unique())

    return frameset_df, n_framesets, dataset_dict


def assign_framesets_to_train_and_valid(
    n_framesets, percent_val, dataset_locs, frameset_df, dataset_dict, save_loc
):
    # subset validation framesets
    n_val_framesets = int(n_framesets * percent_val)
    validation_framesets = np.random.choice(
        np.arange(n_framesets), size=n_val_framesets, replace=False
    )

    # determine which framesets are validation
    frameset_validation = {
        frameset: True if fi in validation_framesets else False
        for fi, frameset in enumerate(frameset_df.frameset.unique())
    }

    # Before the loop iterating over framesets, add a mapping from dataset names to use_in_validation values:
    use_in_validation_mapping = {
        Path(ds["location"]).name: ds["use_in_validation"] for ds in dataset_locs
    }

    # subset validation and training dicts
    annotation_ids_all = [i["id"] for i in dataset_dict["annotations"]]
    annotation_image_ids_all = [i["image_id"] for i in dataset_dict["annotations"]]
    image_ids_all = [i["id"] for i in dataset_dict["images"]]

    val_dict = copy.deepcopy(dataset_dict)
    train_dict = copy.deepcopy(dataset_dict)

    val_dict["images"] = []
    val_dict["annotations"] = []
    val_dict["framesets"] = {}

    train_dict["images"] = []
    train_dict["annotations"] = []
    train_dict["framesets"] = {}
    val_id = 0
    train_id = 0

    # loop through each frameset, placing framesets in training or validation
    for frameset in tqdm(dataset_dict["framesets"].keys(), desc="framesets"):
        # skip any single-image frames
        if len(dataset_dict["framesets"][frameset]["frames"]) == 1:
            continue

        dataset_name = dataset_dict["framesets"][frameset]["datasetName"]

        # determine if this is a validation
        is_validation = frameset_validation[frameset]
        # dont use in validation if this dataset isnt supposed to be used in validation
        if use_in_validation_mapping[dataset_name] == False:
            is_validation = False

        # get the id values of the frameset
        image_id_vals_frameset = dataset_dict["framesets"][frameset]["frames"]

        # get the image dictionaries corresponding to the frameset
        images_idx = [np.where(i == image_ids_all)[0][0] for i in image_id_vals_frameset]
        frameset_images = [copy.deepcopy(dataset_dict["images"][i]) for i in images_idx]

        for frameset_image in frameset_images:
            # add image to training or validation
            if is_validation:
                val_dict["images"].append(frameset_image)
                # move image
                src = save_loc / "train" / frameset_image["file_name"]
                dest = save_loc / "val" / frameset_image["file_name"]
                Path(dest.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
            else:
                train_dict["images"].append(frameset_image)

            ### add annotation to train/valid
            # add each individual annotation
            frameset_annotation_idx = np.where(
                np.array(annotation_image_ids_all) == frameset_image["id"]
            )[0]
            for idx in frameset_annotation_idx:
                if is_validation:
                    val_dict["annotations"].append(dataset_dict["annotations"][idx])
                else:
                    train_dict["annotations"].append(dataset_dict["annotations"][idx])

        if is_validation:
            val_dict["framesets"][frameset] = {
                "datasetName": dataset_dict["framesets"][frameset]["datasetName"],
                "frames": [i["id"] for i in frameset_images],
            }
        else:
            train_dict["framesets"][frameset] = {
                "datasetName": dataset_dict["framesets"][frameset]["datasetName"],
                "frames": [i["id"] for i in frameset_images],
            }
    logger.info(f"Validation samples: {len(val_dict['annotations'])}")
    logger.info(f"Training samples: {len(train_dict['annotations'])}")

    return val_dict, train_dict


def load_jarvis_annotations(annotations_file):
    # get the first four lines, removing the comma
    a_file = open(annotations_file, "r")
    list_of_lines = a_file.readlines()
    a_file.close()
    for i in range(4):
        list_of_lines[i] = "," + list_of_lines[i]
    for i in range(4, len(list_of_lines)):
        list_of_lines[i] = list_of_lines[i][:-2]
    annotations = pd.read_csv(
        io.StringIO(("\n".join(list_of_lines[4:]))), header=None, index_col=0
    )
    header_0 = list_of_lines[0][:-1].split(",")[2:]
    header_1 = list_of_lines[1][:-1].split(",")[2:]
    header_2 = list_of_lines[2][:-1].split(",")[2:]
    header_3 = list_of_lines[3][:-1].split(",")[2:]
    multiindex_col_names = [list_of_lines[i][:-1].split(",")[1] for i in range(4)]
    annotations.columns = pd.MultiIndex.from_tuples(
        (a, b, c, d) for a, b, c, d in zip(header_0, header_1, header_2, header_3)
    ).rename(multiindex_col_names)
    annotations.index = annotations.index.rename("")
    return annotations
