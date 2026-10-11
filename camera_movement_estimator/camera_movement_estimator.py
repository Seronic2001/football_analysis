import cv2
import numpy as np
from utils import draw_translucent_box

class CameraMovementEstimator():
    def __init__(self,frame):
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Feature strips were picked on a 1920px-wide frame; scale to this clip
        scale = first_frame_grayscale.shape[1]/1920
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:max(1,int(20*scale))] = 1
        mask_features[:,int(900*scale):int(1050*scale)] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

        self.old_gray = first_frame_grayscale
        self.old_features = cv2.goodFeaturesToTrack(self.old_gray,**self.features)

    def update(self,frame):
        """Camera movement [x,y] in pixels from the previous frame to this one
        (positive x = camera panned right, so the scene slid left)."""
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        camera_movement = [0.0,0.0]

        if self.old_features is not None:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray,frame_gray,self.old_features,None,**self.lk_params)
            tracked = status.ravel() == 1
            if tracked.any():
                # Median: features that landed on a moving player don't skew it
                flow = (self.old_features - new_features).reshape(-1,2)[tracked]
                camera_movement = np.median(flow,axis=0).tolist()

        self.old_gray = frame_gray
        self.old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)
        return camera_movement

    @staticmethod
    def add_adjust_positions_to_tracks(tracks, camera_movement_per_frame):
        # Per-frame movements accumulate: adding the running total puts every
        # position back in the first frame's coordinates.
        total_movement = np.cumsum(camera_movement_per_frame,axis=0)
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = total_movement[frame_num]
                    position_adjusted = (position[0]+camera_movement[0],position[1]+camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    @staticmethod
    def overlay_frame(frame, camera_movement):
        """Draw one frame's camera readout in place."""
        draw_translucent_box(frame,0,0,500,100,0.6)

        x_movement, y_movement = camera_movement
        frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame
