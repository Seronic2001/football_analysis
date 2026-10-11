import cv2
import numpy as np
from utils import draw_translucent_box

class CameraMovementEstimator():
    """Follows the pitch plane from frame to frame.

    Features come from the grass only (stands, ad boards and detected people
    are masked out) and each step is a RANSAC homography between consecutive
    frames. A broadcast pan rotates the image rather than sliding it - the
    stands and the pitch move in opposite vertical directions - so a single
    translation can't describe it, but a homography of the pitch plane can.
    Chaining the steps gives one matrix per frame that maps it back to the
    first frame's coordinates.
    """
    def __init__(self,frame,boxes=()):
        self.lk_params = dict(
            winSize = (21,21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,20,0.03)
        )
        # Tuned on a 1920px-wide frame; scaled to this clip
        self.scale = frame.shape[1]/1920
        self.features = dict(
            maxCorners = 300,
            qualityLevel = 0.01,
            minDistance = max(3,int(10*self.scale)),
            blockSize = 7,
        )
        self.old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        self.old_mask = self.pitch_mask(frame,boxes)
        self.to_first = np.eye(3)

    def pitch_mask(self,frame,boxes):
        """Grass with the lines filled in, shrunk away from the ad boards, and
        with every detected player, referee and ball cut out."""
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv,(30,40,40),(90,255,255))
        close = max(3,int(31*self.scale))
        grass = cv2.morphologyEx(grass,cv2.MORPH_CLOSE,np.ones((close,close),np.uint8))
        erode = max(3,int(21*self.scale))
        grass = cv2.erode(grass,np.ones((erode,erode),np.uint8))
        pad = int(10*self.scale)
        h, w = grass.shape
        for x1,y1,x2,y2 in boxes:
            grass[max(0,int(y1)-pad):min(h,int(y2)+pad),
                  max(0,int(x1)-pad):min(w,int(x2)+pad)] = 0
        return grass

    def update(self,frame,boxes=()):
        """Returns this frame's 3x3 homography into first-frame coordinates.
        boxes are the frame's detections (x1,y1,x2,y2), kept out of the
        features. A step that can't be estimated is taken as no movement."""
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        step = self.estimate_step(frame_gray)
        if step is not None:
            self.to_first = self.to_first @ step
            self.to_first /= self.to_first[2,2]

        self.old_gray = frame_gray
        self.old_mask = self.pitch_mask(frame,boxes)
        return self.to_first.copy()

    def estimate_step(self,frame_gray):
        """Homography from this frame to the previous one, or None."""
        old_features = cv2.goodFeaturesToTrack(self.old_gray,mask=self.old_mask,**self.features)
        if old_features is None or len(old_features) < 12:
            return None
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(self.old_gray,frame_gray,old_features,None,**self.lk_params)
        tracked = status.ravel() == 1
        if tracked.sum() < 12:
            return None
        step, inliers = cv2.findHomography(new_features[tracked],old_features[tracked],cv2.RANSAC,2.0)
        if step is None or inliers.sum() < 12:
            return None
        # A real step between consecutive frames is small; anything that throws
        # the frame corners far is a bad fit.
        h, w = frame_gray.shape
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        if np.abs(cv2.perspectiveTransform(corners,step)-corners).max() > 0.1*w:
            return None
        return step

    @staticmethod
    def per_frame_movement(to_first,frame_size):
        """Per-frame camera movement [x,y] in pixels at the frame centre, for
        the readout (positive x = camera panned right, so the scene slid left)."""
        w, h = frame_size
        centre = np.float32([[[w/2,h/2]]])
        movement = [[0.0,0.0]]
        for prev, cur in zip(to_first[:-1],to_first[1:]):
            step = np.linalg.inv(prev) @ cur
            moved = cv2.perspectiveTransform(centre,step)
            movement.append((moved-centre).ravel().tolist())
        return movement

    @staticmethod
    def add_adjust_positions_to_tracks(tracks, to_first):
        # Map every position into the first frame's coordinates, which is the
        # frame the ViewTransformer's pitch corners were marked on.
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = np.float32([[track_info['position']]])
                    position_adjusted = cv2.perspectiveTransform(position,to_first[frame_num]).ravel()
                    tracks[object][frame_num][track_id]['position_adjusted'] = tuple(position_adjusted.tolist())

    @staticmethod
    def overlay_frame(frame, camera_movement):
        """Draw one frame's camera readout in place."""
        draw_translucent_box(frame,0,0,500,100,0.6)

        x_movement, y_movement = camera_movement
        frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame
