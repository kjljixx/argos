# Argos
A video-analysis tool for FTC DECODE (and potentially beyond).

## Key Features
From a video of a match, Argos can find:
- How many artifacts each robot scored
- What color artifacts were scored in which order
- For each scored artifact, at exactly what time in the match they were scored
- Analysis can be done in real-time at ~15fps on literally any GPU paired with a decent CPU

## Usage
*Ease of use will be improved later on*
- `git clone` this repo
- Download the robot detection model from https://argos.kjljixx.com/best3.pt
  - Create a `models` folder inside the `argos` directory, and place this model in the `models` folder
- Install `numpy`, `supervision`, `ultralytics`, and `cv2` in your python environment
- Save a match video that you would like to analyze to your computer
  - It's highly recommended to run `ffmpeg -i "original_video.mp4" -vf mpdecimate "deduplicated_video.mp4"` and to use the deduplicated video instead, for more accurate analysis
- Change `PATH` in `polygon_selector.py` to the path to your video
- Run `polygon_selector.py` and create polygons for the goals and launch zones
- - Goal zone polygons should match the shape of the goal opening, like [here](https://github.com/kjljixx/argos/blob/master/docs/images/goal_zone_polygon.png)
  - Launch zone polygons should be made much larger than the actual launch zone, like [here](https://github.com/kjljixx/argos/blob/master/docs/images/launch_zone_polygons.png)
- In `main.py`, Paste the arrays that represent the polygons into `GOAL_ZONES` and `LAUNCH_ZONES` depending on if the polygon is a goal zone or a launch zone
- Change `VIDEO_PATH` in `main.py` to your video
- Run `python3 main.py`
