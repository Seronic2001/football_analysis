import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self,frame_width=1920,frame_height=1080):
        court_width = 68
        court_length = 23.32

        # Pitch corners of the demo camera angle, marked on a 1920x1080 frame;
        # scaled so other resolutions of the same angle still line up.
        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]]) * [frame_width/1920, frame_height/1080]
        
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

        # The four corners only fix the pitch plane; the mapping holds beyond
        # them (the penalty box 35 m outside lands within ~3% of its marked
        # distance), so anything on the pitch is kept. The halfway line sits at
        # x = 5.8 m, so the goal lines of a 105 m pitch are 52.5 m either side.
        halfway_x, half_length, margin = 5.8, 52.5, 3.0
        self.pitch_bounds = ((halfway_x-half_length-margin, halfway_x+half_length+margin),
                             (-margin, court_width+margin))

    def transform_point(self,point):
        """Pitch coordinates in metres, or None off the pitch (stands, or
        above the horizon where the mapping flips)."""
        x, y, w = self.perspective_transformer @ [point[0], point[1], 1.0]
        if w <= 0:
            return None
        x, y = x/w, y/w
        (x_min, x_max), (y_min, y_max) = self.pitch_bounds
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return None
        return np.array([[x, y]])

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed