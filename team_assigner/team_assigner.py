from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        # Fixed seed: identical teams on every run (enables caching & regression checks)
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1, random_state=42)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        """Jersey colour of the player in bbox, or None if the crop is too small to tell."""
        x1,y1,x2,y2 = (max(0,int(v)) for v in bbox)
        image = frame[y1:y2,x1:x2]

        top_half_image = image[0:int(image.shape[0]/2),:]
        if top_half_image.shape[0] < 2 or top_half_image.shape[1] < 2:
            return None

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        """Fit the two team colours from one frame. Returns False (and stays
        unfitted) if the frame doesn't show at least two usable players."""
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color =  self.get_player_color(frame,bbox)
            if player_color is not None:
                player_colors.append(player_color)

        if len(player_colors) < 2:
            return False

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10, random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = tuple(map(float,kmeans.cluster_centers_[0]))
        self.team_colors[2] = tuple(map(float,kmeans.cluster_centers_[1]))
        return True


    def get_player_team(self,frame,player_bbox,player_id):
        """Team (1 or 2) for a player, or None if it can't be decided on this frame yet."""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            return None

        player_color = self.get_player_color(frame,player_bbox)
        if player_color is None:
            return None

        team_id = int(self.kmeans.predict(player_color.reshape(1,-1))[0]) + 1

        self.player_team_dict[player_id] = team_id

        return team_id
