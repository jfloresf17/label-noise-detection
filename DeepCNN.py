from torch import nn

## Define the teacher model
class ModifiedDeepNNSegmenter(nn.Module):
    def __init__(self, num_classes=10):
        super(ModifiedDeepNNSegmenter, self).__init__()

        # Encoder (Feature Extraction)
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x = self.features(x)
        conv_feature_map = x

        # Decoder
        x = self.decoder(x)
        
        return x, conv_feature_map
    