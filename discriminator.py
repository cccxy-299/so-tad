import torch
import torch.nn as nn


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

class Discriminator(nn.Module):
    def __init__(self, input_channel=3, output_channel=32, kernel_size=3, stride=1, bias=False, padding=1):
        super().__init__()
        # c 1 hw
        self.conv1 = Conv3d(input_channel, output_channel * 4,stride=[1,2,2])
        # c 3 hw
        self.conv2 = Conv3d(input_channel, output_channel * 4,stride=[1,2,2])
        output_channel = output_channel * 4
        self.block_one = nn.Sequential(
            nn.Conv3d(output_channel , output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
        )

        self.block_two = nn.Sequential(
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            Residual_Block(output_channel, output_channel),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),

            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(output_channel, output_channel, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),
            nn.Conv3d(output_channel, output_channel, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(output_channel),
            nn.PReLU(),

            nn.Conv3d(output_channel, output_channel, kernel_size, stride, bias=False, padding=1),
        )

        self.output_layer = nn.Sequential(
            nn.Conv3d(output_channel, output_channel * 2 , kernel_size, stride=[2, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(output_channel*2),
            nn.PReLU(),
            nn.Conv3d(output_channel * 2, output_channel * 4, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(output_channel*4),
            nn.PReLU(),

            nn.Conv3d(output_channel * 4, output_channel * 2, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(output_channel*2),
            nn.PReLU(),
        )
        self.linears = nn.Sequential(
            nn.Linear(256 * 5 * 4 , 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )



    def forward(self, input):
        x0 = input[:,:,-1:,:,:]
        x1 = input[:, :, :-1, :, :]
        x0 = self.conv1(x0)
        x1 = self.conv2(x1)

        x0_1 = self.block_one(x0)
        x1_1 = self.block_two(x1)

        x0_1 = self.conv3(x0_1)
        x1_1 = self.conv4(x1_1)

        output = self.output_layer(x0_1 + x1_1)
        output = output.view(-1,256*5*4)
        output = self.linears(output)

        return output

if __name__ == "__main__":
    x = torch.ones([2, 3, 4, 320, 240]).to('cuda')  # batch size ;chanel;

    model = Discriminator().to('cuda')
    a = model(x).view(-1)
    print(a)