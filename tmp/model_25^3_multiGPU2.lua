--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 1:03 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'cunn'
require 'cutorch'
require 'optim'

-- building block
local function ConvBNReLU(base,nInputPlane, nOutputPlane)
    base:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    base:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    base:add(nn.ReLU(true))
    return base
end

local MaxPooling = nn.VolumetricMaxPooling

cutorch.setDevice(1)

layer1 = ConvBNReLU(nn.Sequential(), 1, 64):add(nn.Dropout(0.3));
layer1 = ConvBNReLU(layer1, 64,64);
layer1:add(MaxPooling(2, 2, 2, 2, 2, 2));
layer1:cuda();

cutorch.setDevice(2)
layer2 = ConvBNReLU(nn.Sequential(),1,64):add(nn.Dropout(0.3));
layer2 = ConvBNReLU(layer2, 64,64);
layer2:add(MaxPooling(2, 2, 2, 2, 2, 2));
layer2:cuda();

cutorch.setDevice(3)
-- Bx128x6x6x6
layer3 = ConvBNReLU(nn.Sequential(), 128, 256):add(nn.Dropout(0.4))
layer3 = ConvBNReLU(layer3,256, 256):add(nn.Dropout(0.4))
layer3 = ConvBNReLU(layer3,256, 256)
layer3:add(MaxPooling(2, 2, 2, 2, 2, 2))
layer3:cuda();

cutorch.setDevice(4)
-- Bx256x3x3x3
layer4 = ConvBNReLU(nn.Sequential(), 256, 512):add(nn.Dropout(0.4))
layer4 = ConvBNReLU(layer4, 512, 512):add(nn.Dropout(0.4))
layer4 = ConvBNReLU(layer4, 512, 512)
layer4:add(MaxPooling(2, 2, 2, 2, 2, 2));
layer4:add(nn.View(512))

-- Bx512x1x1x1
classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 2))
layer4:add(classifier)
layer4:cuda();

cutorch.setDevice(1)
input = torch.randn(1,1,20,256,256):cuda();

cutorch.setDevice(4)
target = torch.randn(1,1,20,256,256):cuda()
criterion = nn.CrossEntropyCriterion():cuda()

for i=1,1000 do
    print('in iteration:',i)
    cutorch.setDevice(1)
    output1 = layer1:forward(input);
    cutorch.setDevice(2)
    output2 = layer2:forward(outpu1);

    cutorch.setDevice(3)
    output3 = layer3:forward(outpu2);

    cutorch.setDevice(4)
    output4 = layer4:forward(outpu3);


    err = criterion:forward(output4, target)
    df_do = criterion:backward(output4, target)
    gradInput4 = layer4:backward(output4, df_do)
    cutorch.setDevice(3)
    gradInput3 = layer3:backward(output3, gradInput4)
    cutorch.setDevice(2)
    gradInput2 = layer2:backward(output2, gradInput3)
    cutorch.setDevice(1)
    gradInput1 = layer1:backward(output1, gradInput2)


end



-- initialization from MSR
--local function MSRinit(net)
--    local function init(name)
--        for k, v in pairs(net:findModules(name)) do
--            local n = v.kW * v.kH * v.nOutputPlane
--            v.weight:normal(0, math.sqrt(2 / n))
--            v.bias:zero()
--        end
--    end
--
--    -- have to do for both backends
--    init 'nn.SpatialConvolution'
--end
--
--MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x2 tensor
--print(vgg:cuda():forward(torch.CudaTensor(16,1,25,25,25)))

return layer1,layer2,layer3,layer4;