#   Partial implementation of:
#       Huang & Belongie, "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization", arXiv:1703.06868v2, 30 July 2017
#
#   Adapted from:
#       https://github.com/naoto0804/pytorch-AdaIN
#

import torch.nn as nn

class encoder_decoder:
    encoder = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
class AdaIN_net(nn.Module):

    def __init__(self, encoder, decoder=None):
        super(AdaIN_net, self).__init__()
        self.encoder = encoder
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # need access to these intermediate encoder steps
        # for the AdaIN computation
        encoder_list = list(encoder.children())
        self.encoder_stage_1 = nn.Sequential(*encoder_list[:4])  # input -> relu1_1
        self.encoder_stage_2 = nn.Sequential(*encoder_list[4:11])  # relu1_1 -> relu2_1
        self.encoder_stage_3 = nn.Sequential(*encoder_list[11:18])  # relu2_1 -> relu3_1
        self.encoder_stage_4 = nn.Sequential(*encoder_list[18:31])  # relu3_1 -> relu4_1

        self.decoder = decoder
        #   if no decoder loaded, then initialize with random weights
        if self.decoder == None:
            # self.decoder = _decoder
            self.decoder = encoder_decoder.decoder
            self.init_decoder_weights(mean=0.0, std=0.01)

        self.mse_loss = nn.MSELoss()

    def init_decoder_weights(self, mean, std):
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)
    def encode(self, X):
        relu1_1 = self.encoder_stage_1(X)
        relu2_1 = self.encoder_stage_2(relu1_1)
        relu3_1 = self.encoder_stage_3(relu2_1)
        relu4_1 = self.encoder_stage_4(relu3_1)
        return relu1_1, relu2_1, relu3_1, relu4_1

    def decode(self, X):
        return(self.decoder(X))

    #   Eq. (12)
    def content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    #   Eq. (13)
    def style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = self.mean_std(input)
        target_mean, target_std = self.mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    #   Eq. (8)
    def adain(self, content_feat, style_feat):
        assert (content_feat.size()[:2] == style_feat.size()[:2])
        size = content_feat.size()
        style_mean, style_std = self.mean_std(style_feat)
        content_mean, content_std = self.mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(
            size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std


    # NOTE: forward was derived using the forward method found in https://github.com/naoto0804/pytorch-AdaIN/blob/master/net.py
    def forward(self, content, style, alpha=1.0):
        # If alpha is not within the allowable range, throw an 'AssertionError' exception
        assert 0 <= alpha <= 1

        # Encode the content image using the VGG encoder
        content_features = self.encode(content)[-1]
        # Encode the style image using the VGG encoder
        style_features = self.encode(style)

        # Apply the AdaIN transformation
        transformed_features = self.adain(content_features, style_features[-1])
        transformed_features = alpha * transformed_features + (1 - alpha) * content_features

        # Decode the transformed features to produce the generated image
        generated_image = self.decode(transformed_features)

        if self.training:  # Training
            # Encode the generated image using the VGG encoder
            generated_image_features = self.encode(generated_image)

            # Calculate the content loss using Eq. (12)
            loss_c = self.content_loss(generated_image_features[-1], transformed_features)
            # Calculate the style loss using Eq. (13) for the first layer
            loss_s = self.style_loss(generated_image_features[0], style_features[0])
            # Calculate the style loss using Eq. (13) for layers 2 to 4
            for i in range(1, 4):
                loss_s += self.style_loss(generated_image_features[i], style_features[i])

            # Return the content loss and style loss
            return loss_c, loss_s
        else:  # Inference
            # Return the generated image
            return generated_image
