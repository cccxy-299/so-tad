
import torch
import torch.nn as nn


class OutputConv3d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),

            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
        )
    def forward(self, x0):
        x1 = self.layer(x0)
        return x1

class Conv3d(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),

            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel)
        )
    def forward(self, x0):
        x1 = self.layer(x0)
        return x1

class Residual_Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(input_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),

            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel)
        )
    def forward(self, x0):
        x1 = self.layer(x0)
        return x0 + x1

class Generator(nn.Module):
    def __init__(self,input_channel=3, output_channel=32, kernel_size=3, stride=1, bias=False, padding=1):
        super().__init__()

        self.conv1 = Conv3d(input_channel, output_channel)
        self.conv2 = Conv3d(input_channel, output_channel)
        self.conv3 = Conv3d(input_channel, output_channel)

        self.block_one = nn.Sequential(
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            Residual_Block(output_channel,output_channel),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
        )
        #
        self.block_two = nn.Sequential(
            nn.Conv3d(output_channel, output_channel*2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel*2),
            nn.PReLU(),
            Residual_Block(output_channel * 2, output_channel * 2),
            Residual_Block(output_channel * 2, output_channel * 2),
            Residual_Block(output_channel * 2, output_channel * 2),
            Residual_Block(output_channel * 2, output_channel * 2),

            nn.Conv3d(output_channel * 2, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
        )
        #
        self.block_three = nn.Sequential(
            nn.Conv3d(output_channel, output_channel * 3, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel * 3),
            nn.PReLU(),
            Residual_Block(output_channel * 3, output_channel * 3),
            Residual_Block(output_channel * 3, output_channel * 3),
            Residual_Block(output_channel * 3, output_channel * 3),
            Residual_Block(output_channel * 3, output_channel * 3),

            nn.Conv3d(output_channel * 3, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
        )

        self.conv4 = Conv3d(output_channel, output_channel)
        self.conv5 = Conv3d(output_channel, output_channel)
        self.conv6 = Conv3d(output_channel, output_channel)

        self.conv7 = nn.Sequential(
            nn.Conv3d(output_channel, 3, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(3),
            nn.PReLU(),

            nn.Conv3d(3, 3, kernel_size, stride, bias=False, padding=1),
        )
        self.conv8 = nn.Sequential(
            nn.Conv3d(output_channel, 3, kernel_size, stride=[2,1,1], bias=False, padding=1),
            nn.BatchNorm3d(3),
            nn.PReLU(),

            nn.Conv3d(3, 3, kernel_size, stride, bias=False, padding=1),
        )
        self.conv9 = nn.Sequential(
            nn.Conv3d(output_channel, 3, kernel_size,  stride=[2,1,1], bias=False, padding=1),
            nn.BatchNorm3d(3),
            nn.PReLU(),

            nn.Conv3d(3, 3, kernel_size, stride=[2,1,1], bias=False, padding=1),
        )

        self.conv10 = OutputConv3d(3, 3)

    def forward(self, frames):
        frames = frames
        x1 = frames[:,:,-1:,:,:]
        x2 = frames[:, :, -2:, :, :]
        x3 = frames
        x_1 = self.conv1(x1)
        x_2 = self.conv2(x2)
        x_3 = self.conv3(x3)

        #
        x1 = self.block_one(x_1)
        x2 = self.block_two(x_2)
        x3 = self.block_three(x_3)

        #
        x1 = self.conv4(x_1+x1)
        x2 = self.conv5(x_2+x2)
        x3 = self.conv6(x_3+x3)

        x1 = self.conv7(x1)
        x2 = self.conv8(x2)
        x3 = self.conv9(x3)


        pred = self.conv10(x1+x2+x3)

        return pred, x1, x2, x3

if __name__ == "__main__":
    x = torch.ones([2, 3, 4, 320, 240]).to('cuda')
    gan = Generator().to('cuda')

    a = gan(x)[0]
    print(a.shape)