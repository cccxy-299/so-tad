import torch
import torch.nn as nn



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
        return x0+x1

class Generator(nn.Module):
    def __init__(self,input_channel, ch, output_channel, upscale_factor=4, kernel_size=3, stride=1, bias=False, padding=1):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

        self.conv_in_1 = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch*2, kernel_size, stride=[1,2,2], bias=False, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.PReLU(),
        )

        self.conv_in_2 = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            nn.Conv3d(ch * 2, ch*2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.PReLU(),
        )
        #
        self.conv_in_3 = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            nn.Conv3d(ch * 2, ch*2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.PReLU(),
        )
        #
        self.pipeline_1 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            # Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
        )
        #
        self.pipeline_2 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            # Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
        )
        #
        self.pipeline_3 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            # Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 2, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
        )

        self.conv_in_4 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[1, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.conv_in_5 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.conv_in_6 = nn.Sequential(
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2, 2, 2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride=[2, 1, 1], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.act_fnc = nn.PReLU()

        self.conv_in_7 = nn.Sequential(
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
            nn.PReLU(),
            nn.Conv3d(ch * 4, ch * 4, kernel_size, stride, bias=False, padding=1),
        )

    def forward(self, t0, t1, t2, d01, d12):
        # [2, 256, 37, 27] -> [2, 16, 1,148, 108]
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        d01 = torch.unsqueeze(self.ps(d01), 2)
        d12 = torch.unsqueeze(self.ps(d12), 2)

        # [2, 16, 1,148, 108] -> [2, 16, 1,148, 108]
        x0 = t0
        # [2, 16, 1,148, 108] -> [2, 16, 3,148, 108]
        x1 = torch.concatenate((t0, d01, t1),dim=2)
        # [2, 16, 1,148, 108] -> [2, 16, 5,148, 108]
        x2 = torch.concatenate((t0, d01, t1, d12, t2), dim=2)

        # 流程1
        #  # [2, 16, 1,148, 108] -> [2, 256, 1,37, 27]
        x0 = self.conv_in_1(x0)
        x00 = self.pipeline_1(x0)
        x00 = self.act_fnc(x0+x00)
        x00 = self.conv_in_4(x00)

        # 流程2
        # [2, 16, 1,148, 108] -> [2, 256, 1,37, 27]
        x1 = self.conv_in_2(x1)
        x11 = self.pipeline_2(x1)
        x11 = self.act_fnc(x1 + x11)
        x11 = self.conv_in_5(x11)

        # 流程3
        # [2, 16, 1,148, 108] -> [2, 256, 1,37, 27]
        x2 = self.conv_in_3(x2)
        x22 = self.pipeline_3(x2)
        x22 = self.act_fnc(x2 + x22)
        x22 = self.conv_in_6(x22)

        # [2, 256, 1,37, 27] -> [2, 256, 1,37, 27]
        output = self.act_fnc(x00 + x11 + x22)
        output = self.conv_in_7(output)
        output = torch.squeeze(output, 2)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_channel, ch, output_channel, upscale_factor=4,kernel_size=3, stride=1, bias=False, padding=1):
        super().__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

        self.convs_3d = nn.Sequential(
            nn.Conv3d(input_channel, input_channel, kernel_size, stride, bias=False, padding=1),
            nn.BatchNorm3d(input_channel),
            nn.PReLU(),
            nn.Conv3d(input_channel, ch * 2, kernel_size, stride=[2,2,2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 2),
            nn.PReLU(),
            Residual_Block(ch * 2, ch * 2),
            # Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            Residual_Block(ch * 2, ch * 2),
            nn.Conv3d(ch * 2, ch * 4, kernel_size, stride=[2,2,2], bias=False, padding=1),
            nn.BatchNorm3d(ch * 4),
        )

        self.convs_2d = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(ch * 8),
            nn.PReLU(),
            nn.Conv2d(ch * 8, ch * 16, kernel_size, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 16, kernel_size, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(ch * 16),
            nn.PReLU(),
            nn.Conv2d(ch * 16, ch * 32, kernel_size, stride=2, bias=False, padding=1),
        )

        self.act_fnc = nn.PReLU()

        self.linears = nn.Sequential(
            nn.Linear(ch * 32 * 5 * 4, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, t0, t1, t2, t3):
        t0 = torch.unsqueeze(self.ps(t0), 2)
        t1 = torch.unsqueeze(self.ps(t1), 2)
        t2 = torch.unsqueeze(self.ps(t2), 2)
        t3 = torch.unsqueeze(self.ps(t3), 2)

        x0 = torch.concatenate((t0, t1, t2, t3), dim=2)
        x1 = self.convs_3d(x0)
        x1 = self.act_fnc(x1)
        x1 = torch.squeeze(x1, 2)
        x1 = self.convs_2d(x1)
        x1 = x1.view(-1,512*5*4)
        output = self.linears(x1)


        return output


if __name__ == "__main__":
    t0 = torch.ones([2, 256, 77, 57]).to('cuda')
    t1 = torch.ones([2, 256, 77, 57]).to('cuda')
    t2 = torch.ones([2, 256, 77, 57]).to('cuda')
    d01 = torch.ones([2, 256, 77, 57]).to('cuda')
    d12 = torch.ones([2, 256, 77, 57]).to('cuda')
    model = Discriminator(16, 16, 16).to('cuda')
    model1 = Discriminator(16, 16, 16).to('cuda')

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    total_params1 = sum(p.numel() for p in model1.parameters())
    print(total_params1)
    #
    a = model(t0, t1, t2, d01)
    # a = torch.squeeze(a,2)
    # print(a.shape)