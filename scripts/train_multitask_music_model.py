class MultiTaskMusicModel(nn.Module):

    def __init__(

        self,

        num_genres
    ):

        super().__init__()

        self.backbone = models.resnet18(
            pretrained=True
        )

        # ---------------------------------
        # Shared Feature Extractor
        # ---------------------------------

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        # ---------------------------------
        # Genre Head
        # ---------------------------------

        self.genre_head = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(

                in_features,

                num_genres
            )
        )

        # ---------------------------------
        # Valence Head
        # ---------------------------------

        self.valence_head = nn.Sequential(

            nn.Linear(
                in_features,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            )
        )

        # ---------------------------------
        # Arousal Head
        # ---------------------------------

        self.arousal_head = nn.Sequential(

            nn.Linear(
                in_features,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):

        features = self.backbone(x)

        genre_logits = (
            self.genre_head(features)
        )

        valence = (
            self.valence_head(features)
        )

        arousal = (
            self.arousal_head(features)
        )

        return (

            genre_logits,

            valence,

            arousal
        )
