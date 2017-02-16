--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 1:03 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'cunn'
require './util'

vgg = nn.Sequential();

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
    vgg:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    vgg:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.ReLU(true))
    return vgg
end

local function ConvBNReLUStride2(convNet,nInputPlane, nOutputPlane)
    convNet:add(nn.VolumetricConvolution(nInputPlane, nOutputPlane, 5, 5, 5, 1, 1, 1, 1, 1, 1))
    convNet:add(nn.VolumetricBatchNormalization(nOutputPlane, 1e-3))
    convNet:add(nn.ReLU(true))
    return convNet
end

local MaxPooling = nn.VolumetricMaxPooling
-- Bx1x25x25x25
ConvBNReLUStride2(vgg,1,64);
ConvBNReLUStride2(vgg,64,64);
-- Bx64x23x23x23

--ConvBNReLU(1,64):add(nn.Dropout(0.3));
--ConvBNReLU(64,64);

vgg:add(MaxPooling(2, 2, 2, 2, 2, 2));

-- Bx64x11x11x11
ConvBNReLU(64, 128):add(nn.Dropout(0.4))
ConvBNReLU(128, 128)
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2))

-- Bx128x5x5x5
ConvBNReLU(128, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256)
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2))

-- Bx256x2x2x2
ConvBNReLU(256, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
vgg:add(MaxPooling(2, 2, 2, 2, 2, 2))


ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)

-- Bx512x1x1x1
vgg:add(nn.View(512))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 512))
classifier:add(nn.BatchNormalization(512))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512, 2))
vgg:add(classifier)


--gpus = torch.range(1, cutorch.getDeviceCount()):totable()
--dpt = nn.DataParallelTable(1,true,true):add(vgg, gpus):cuda()

dpt = makeDataParallel(vgg,cutorch.getDeviceCount());


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
print(dpt:cuda():forward(torch.CudaTensor(16,1,25,25,25)):size())
getMemStats();
collectgarbage();

return dpt